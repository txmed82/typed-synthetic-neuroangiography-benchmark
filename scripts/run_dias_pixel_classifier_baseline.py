#!/usr/bin/env python3
"""Learned DIAS pixel-classifier vessel baseline.

This is not nnU-Net. It is a local CPU learned-control baseline that uses DIAS
training labels and per-pixel temporal projection features. It is intended to set
a materially stronger reference than global threshold/morphology wiring baselines
without GPU infrastructure.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = ROOT / "scripts" / "run_dias_segmentation_baseline.py"
spec = importlib.util.spec_from_file_location("run_dias_segmentation_baseline", BASELINE_SCRIPT)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)  # type: ignore[union-attr]


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def record_feature_image(record: dict[str, Any], root: Path) -> np.ndarray:
    frame_paths = baseline.record_frame_paths(record, root)
    stack = np.stack([load_gray(path) for path in frame_paths], axis=0)
    max_img = stack.max(axis=0)
    min_img = stack.min(axis=0)
    mean_img = stack.mean(axis=0)
    range_img = max_img - min_img
    std_img = stack.std(axis=0)
    h, w = max_img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x_norm = xx.astype(np.float32) / max(1, w - 1)
    y_norm = yy.astype(np.float32) / max(1, h - 1)
    return np.stack([max_img, min_img, mean_img, range_img, std_img, x_norm * 255.0, y_norm * 255.0], axis=-1)


def record_truth(record: dict[str, Any], root: Path) -> np.ndarray:
    lp = baseline.label_path(record, root)
    if lp is None:
        raise ValueError(f"no label path for {record['sequence_id']}")
    return load_gray(lp) > 0


def sample_training_pixels(records: list[dict[str, Any]], root: Path, per_class_per_record: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for record in records:
        feat = record_feature_image(record, root).reshape(-1, 7)
        truth = record_truth(record, root).reshape(-1)
        pos = np.flatnonzero(truth)
        neg = np.flatnonzero(~truth)
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos_idx = rng.choice(pos, size=min(per_class_per_record, len(pos)), replace=len(pos) < per_class_per_record)
        neg_idx = rng.choice(neg, size=min(per_class_per_record, len(neg)), replace=len(neg) < per_class_per_record)
        idx = np.concatenate([pos_idx, neg_idx])
        xs.append(feat[idx])
        ys.append(truth[idx].astype(np.uint8))
    if not xs:
        raise ValueError("no training pixels sampled")
    return np.vstack(xs), np.concatenate(ys)


def probability_mask(record: dict[str, Any], root: Path, clf: RandomForestClassifier, threshold: float, min_component_area: int) -> Image.Image:
    feat_img = record_feature_image(record, root)
    h, w, c = feat_img.shape
    probs = clf.predict_proba(feat_img.reshape(-1, c))[:, 1].reshape(h, w)
    mask = Image.fromarray(np.where(probs >= threshold, 255, 0).astype(np.uint8), mode="L")
    return baseline.remove_small_components(mask, min_area=min_component_area)


def evaluate(records: list[dict[str, Any]], root: Path, clf: RandomForestClassifier, threshold: float, min_component_area: int, pred_dir: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        pred = probability_mask(record, root, clf, threshold, min_component_area)
        truth_img = Image.open(baseline.label_path(record, root)).convert("L")  # type: ignore[arg-type]
        metrics = baseline.binary_image_metrics(truth_img, pred)
        pred_uri = None
        if pred_dir:
            pred_path = pred_dir / f"{record['sequence_id']}_rf_pixel_mask.png"
            pred.save(pred_path)
            pred_uri = str(pred_path)
        rows.append({
            "sequence_id": record["sequence_id"],
            "split": record["split"],
            "model_name": "dias_random_forest_pixel_classifier",
            "probability_threshold": threshold,
            "min_component_area": min_component_area,
            "prediction_uri": pred_uri,
            **metrics,
        })
        truth_img.close()
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_iou": mean([float(r["iou"]) for r in rows]) if rows else 0.0,
        "mean_dice": mean([float(r["dice"]) for r in rows]) if rows else 0.0,
        "min_iou": min([float(r["iou"]) for r in rows]) if rows else 0.0,
        "min_dice": min([float(r["dice"]) for r in rows]) if rows else 0.0,
        "mean_truth_area_px": mean([int(r["truth_area_px"]) for r in rows]) if rows else 0.0,
        "mean_pred_area_px": mean([int(r["pred_area_px"]) for r in rows]) if rows else 0.0,
    }


def tune_validation(validation: list[dict[str, Any]], root: Path, clf: RandomForestClassifier) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        for min_component_area in [0, 16, 64, 256, 1024]:
            rows = evaluate(validation, root, clf, threshold, min_component_area, pred_dir=None)
            agg = aggregate(rows)
            cand = {"probability_threshold": threshold, "min_component_area": min_component_area, "validation_aggregate": agg}
            if best is None or agg["mean_dice"] > best["validation_aggregate"]["mean_dice"]:
                best = cand
    assert best is not None
    return best


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# DIAS random-forest pixel-classifier baseline",
        "",
        "This is a local CPU learned-control baseline, not nnU-Net or a published DIAS SOTA model.",
        "It uses DIAS training labels and temporal projection features for each pixel.",
        "",
        f"Train sequences: {report['train_sequence_count']}",
        f"Validation sequences: {report['validation_sequence_count']}",
        f"Test sequences: {report['test_sequence_count']}",
        f"Validation-selected probability threshold: {report['model']['probability_threshold']}",
        f"Validation-selected min component area: {report['model']['min_component_area']}",
        "",
        "## Validation aggregate",
        "",
    ]
    for k, v in report["validation_aggregate"].items():
        lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    lines.extend(["", "## Test aggregate", ""])
    for k, v in report["test_aggregate"].items():
        lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_jsonl", type=Path, default=ROOT / "outputs/manifests/dias_manifest.jsonl")
    ap.add_argument("--dataset-root", type=Path, default=ROOT / "data/dias/DIAS")
    ap.add_argument("--sample-per-class-per-record", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--out-json", type=Path, default=ROOT / "outputs/reports/dias_random_forest_pixel_classifier_report.json")
    ap.add_argument("--out-md", type=Path, default=ROOT / "outputs/reports/dias_random_forest_pixel_classifier_report.md")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "outputs/dias_predictions/random_forest_pixel_classifier")
    args = ap.parse_args()

    records = baseline.load_manifest(args.manifest_jsonl)
    labeled = [r for r in records if r.get("has_labels")]
    train = [r for r in labeled if r.get("split") == "training"]
    validation = [r for r in labeled if r.get("split") == "validation"]
    test = [r for r in labeled if r.get("split") == "test"]
    x_train, y_train = sample_training_pixels(train, args.dataset_root, args.sample_per_class_per_record, args.seed)
    clf = RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=8, class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed)
    clf.fit(x_train, y_train)
    tuned = tune_validation(validation, args.dataset_root, clf)
    threshold = float(tuned["probability_threshold"])
    min_component_area = int(tuned["min_component_area"])
    validation_rows = evaluate(validation, args.dataset_root, clf, threshold, min_component_area, args.pred_dir / "validation")
    test_rows = evaluate(test, args.dataset_root, clf, threshold, min_component_area, args.pred_dir / "test")
    report = {
        "dataset": "DIAS",
        "manifest": str(args.manifest_jsonl),
        "dataset_root": str(args.dataset_root),
        "model": {
            "name": "dias_random_forest_pixel_classifier",
            "version": "0.1.0",
            "feature_set": ["temporal_max", "temporal_min", "temporal_mean", "temporal_range", "temporal_std", "x_norm", "y_norm"],
            "estimator": "sklearn RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=8, class_weight=balanced_subsample)",
            "sample_per_class_per_training_record": args.sample_per_class_per_record,
            "probability_threshold": threshold,
            "min_component_area": min_component_area,
        },
        "train_sequence_count": len(train),
        "validation_sequence_count": len(validation_rows),
        "test_sequence_count": len(test_rows),
        "validation_aggregate": aggregate(validation_rows),
        "test_aggregate": aggregate(test_rows),
        "validation_per_sequence": validation_rows,
        "test_per_sequence": test_rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, args.out_md)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "validation": report["validation_aggregate"], "test": report["test_aggregate"], "model": report["model"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
