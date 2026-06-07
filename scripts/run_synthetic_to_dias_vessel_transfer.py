#!/usr/bin/env python3
"""Synthetic-to-DIAS vessel segmentation relevance experiment.

This experiment intentionally stays dependency-light. It uses synthetic Seldinger-DSA
variants to estimate vessel occupancy priors, applies those priors as an adaptive
per-sequence top-percentile rule on DIAS temporal-range projections, and compares
against the existing DIAS projection-threshold/projection-morphology baselines.

Scope: vessel masks only. DIAS has no catheter/device labels.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = ROOT / "scripts" / "run_dias_segmentation_baseline.py"
spec = importlib.util.spec_from_file_location("run_dias_segmentation_baseline", BASELINE_SCRIPT)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)  # type: ignore[union-attr]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve(root: Path, uri: str) -> Path:
    p = Path(uri)
    return p if p.is_absolute() else root / p


def union_mask_area_percent(record: dict[str, Any], root: Path) -> float:
    """Return percent of pixels covered by the union of a synthetic mask sequence."""
    mask_info = record["vessel_mask_sequence"]
    mask_path = resolve(root, mask_info["uri"])
    frame_count = int(mask_info.get("frame_count", 1))
    if mask_path.is_file():
        img = Image.open(mask_path).convert("L")
        data = img.tobytes()
        return 100.0 * sum(v > 0 for v in data) / len(data)

    union: bytearray | None = None
    size: tuple[int, int] | None = None
    for i in range(frame_count):
        img = Image.open(mask_path / f"mask_{i:03d}.png").convert("L")
        if union is None:
            size = img.size
            union = bytearray(len(img.tobytes()))
        elif img.size != size:
            raise ValueError(f"mask size mismatch for {record['sequence_id']}")
        for idx, value in enumerate(img.tobytes()):
            if value:
                union[idx] = 1
    if union is None:
        raise ValueError(f"no masks found for {record['sequence_id']}")
    return 100.0 * sum(union) / len(union)


def synthetic_area_prior(manifest_path: Path, root: Path) -> dict[str, Any]:
    records = load_jsonl(manifest_path)
    pcts = [union_mask_area_percent(r, root) for r in records]
    return {
        "manifest": str(manifest_path),
        "sequence_count": len(pcts),
        "mean_area_percent": mean(pcts),
        "median_area_percent": median(pcts),
        "min_area_percent": min(pcts),
        "max_area_percent": max(pcts),
    }


def top_percent_mask(proj: Image.Image, area_percent: float) -> Image.Image:
    """Keep the highest temporal-range pixels according to an area prior."""
    hist = proj.histogram()
    total = proj.size[0] * proj.size[1]
    target = max(1, int(total * area_percent / 100.0))
    running = 0
    threshold = 255
    for value in range(255, -1, -1):
        running += hist[value]
        if running >= target:
            threshold = value
            break
    return baseline.threshold_mask_image(proj, threshold, polarity=">=")


def evaluate_area_prior(
    dias_records: list[dict[str, Any]],
    dias_root: Path,
    split: str,
    model_name: str,
    area_percent: float,
    pred_dir: Path | None,
    min_component_area: int = 16,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)
    for record in dias_records:
        if record.get("split") != split or not record.get("has_labels"):
            continue
        proj = baseline.projection_image(baseline.record_frame_paths(record, dias_root), mode="range")
        pred = top_percent_mask(proj, area_percent)
        pred = baseline.remove_small_components(pred, min_area=min_component_area)
        truth = Image.open(resolve(dias_root, record["vessel_mask_sequence"]["uri"])).convert("L")
        metrics = baseline.binary_image_metrics(truth, pred)
        pred_uri = None
        if pred_dir:
            pred_path = pred_dir / f"{record['sequence_id']}_range_{model_name}_mask.png"
            pred.save(pred_path)
            pred_uri = str(pred_path)
        rows.append({
            "sequence_id": record["sequence_id"],
            "split": split,
            "frame_count": record["dsa_frame_sequence"]["frame_count"],
            "model_name": model_name,
            "projection": "range",
            "area_percent": area_percent,
            "min_component_area": min_component_area,
            "prediction_uri": pred_uri,
            **metrics,
        })
        truth.close()
    return {
        "model_name": model_name,
        "split": split,
        "area_percent": area_percent,
        "min_component_area": min_component_area,
        "aggregate": baseline.aggregate(rows),
        "per_sequence": rows,
    }


def load_baseline_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Synthetic-to-DIAS vessel segmentation relevance experiment",
        "",
        "Scope: vessel masks only. DIAS does not provide catheter/device labels.",
        "",
        "## Synthetic vessel-area priors",
        "",
    ]
    for key, prior in report["synthetic_priors"].items():
        lines.extend([
            f"- {key}: mean={prior['mean_area_percent']:.3f}% median={prior['median_area_percent']:.3f}% min={prior['min_area_percent']:.3f}% max={prior['max_area_percent']:.3f}% sequences={prior['sequence_count']}",
        ])
    lines.extend(["", "## DIAS comparison", ""])
    for row in report["comparison"]:
        lines.append(
            f"- {row['model']}: validation Dice={row.get('validation_dice', 0):.4f}, IoU={row.get('validation_iou', 0):.4f}; "
            f"test Dice={row.get('test_dice', 0):.4f}, IoU={row.get('test_iou', 0):.4f}"
        )
    lines.extend([
        "",
        "## Readout",
        "",
        f"- Best synthetic-prior variant by validation Dice: `{report['best_synthetic_by_validation']['model_name']}`.",
        f"- Test Dice delta versus DIAS projection-threshold baseline: {report['best_synthetic_by_validation']['delta_vs_test_threshold_dice']:+.4f}.",
        f"- Test Dice delta versus DIAS projection-morphology baseline: {report['best_synthetic_by_validation']['delta_vs_test_morphology_dice']:+.4f}.",
        "- This supports a limited synthetic-to-real relevance claim for vessel-mask stress/augmentation only, not catheter/device transfer.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_experiment(
    dias_manifest: Path,
    dias_root: Path,
    synthetic_manifests: list[Path],
    synthetic_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    dias_records = load_jsonl(dias_manifest)
    priors = {p.stem.replace("_train42_manifest", ""): synthetic_area_prior(p, synthetic_root) for p in synthetic_manifests}

    results: dict[str, dict[str, Any]] = {}
    for variant, prior in priors.items():
        model_name = f"synthetic_area_prior_{variant}"
        area = float(prior["mean_area_percent"])
        for split in ["validation", "test"]:
            pred_dir = out_dir / "dias_predictions" / model_name / split
            results[f"{model_name}:{split}"] = evaluate_area_prior(
                dias_records,
                dias_root,
                split=split,
                model_name=model_name,
                area_percent=area,
                pred_dir=pred_dir,
                min_component_area=16,
            )

    baseline_paths = {
        "dias_projection_threshold_validation": synthetic_root / "outputs/reports/dias_validation_projection_threshold_report.json",
        "dias_projection_threshold_test": synthetic_root / "outputs/reports/dias_test_projection_threshold_report.json",
        "dias_projection_morphology_validation": synthetic_root / "outputs/reports/dias_validation_projection_morphology_report.json",
        "dias_projection_morphology_test": synthetic_root / "outputs/reports/dias_test_projection_morphology_report.json",
    }
    baselines = {k: load_baseline_report(v) for k, v in baseline_paths.items()}

    comparison: list[dict[str, Any]] = []
    def add_comparison(model: str, val: dict[str, Any] | None, test: dict[str, Any] | None) -> None:
        comparison.append({
            "model": model,
            "validation_dice": None if val is None else val["aggregate"]["mean_dice"],
            "validation_iou": None if val is None else val["aggregate"]["mean_iou"],
            "test_dice": None if test is None else test["aggregate"]["mean_dice"],
            "test_iou": None if test is None else test["aggregate"]["mean_iou"],
        })

    add_comparison("DIAS projection-threshold", baselines["dias_projection_threshold_validation"], baselines["dias_projection_threshold_test"])
    add_comparison("DIAS projection-morphology", baselines["dias_projection_morphology_validation"], baselines["dias_projection_morphology_test"])
    for variant in priors:
        model_name = f"synthetic_area_prior_{variant}"
        add_comparison(model_name, results[f"{model_name}:validation"], results[f"{model_name}:test"])

    synthetic_rows = [r for r in comparison if r["model"].startswith("synthetic_area_prior_")]
    best = max(synthetic_rows, key=lambda r: float(r["validation_dice"]))
    threshold_test = baselines["dias_projection_threshold_test"]["aggregate"]["mean_dice"] if baselines["dias_projection_threshold_test"] else 0.0
    morphology_test = baselines["dias_projection_morphology_test"]["aggregate"]["mean_dice"] if baselines["dias_projection_morphology_test"] else 0.0
    best_detail = {
        "model_name": best["model"],
        "validation_dice": best["validation_dice"],
        "test_dice": best["test_dice"],
        "delta_vs_test_threshold_dice": float(best["test_dice"]) - float(threshold_test),
        "delta_vs_test_morphology_dice": float(best["test_dice"]) - float(morphology_test),
    }

    return {
        "experiment": "synthetic_to_dias_vessel_transfer_v0",
        "scope": "vessel segmentation only; DIAS has no catheter/device labels",
        "dias_manifest": str(dias_manifest),
        "dias_root": str(dias_root),
        "synthetic_priors": priors,
        "baseline_reports": {k: str(v) for k, v in baseline_paths.items()},
        "results": results,
        "comparison": comparison,
        "best_synthetic_by_validation": best_detail,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias-manifest", type=Path, default=ROOT / "outputs/manifests/dias_manifest.jsonl")
    ap.add_argument("--dias-root", type=Path, default=ROOT / "data/dias/DIAS")
    ap.add_argument("--synthetic-manifest", action="append", type=Path, dest="synthetic_manifests")
    ap.add_argument("--out-json", type=Path, default=ROOT / "outputs/reports/synthetic_to_dias_vessel_transfer_report.json")
    ap.add_argument("--out-md", type=Path, default=ROOT / "outputs/reports/synthetic_to_dias_vessel_transfer_report.md")
    args = ap.parse_args()

    synthetic_manifests = args.synthetic_manifests or [
        ROOT / "outputs/manifests/toy_v2_train42_manifest.jsonl",
        ROOT / "outputs/manifests/toy_v3_train42_manifest.jsonl",
        ROOT / "outputs/manifests/toy_v4_train42_manifest.jsonl",
    ]
    report = run_experiment(args.dias_manifest, args.dias_root, synthetic_manifests, ROOT, ROOT / "outputs")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, args.out_md)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "best": report["best_synthetic_by_validation"], "comparison": report["comparison"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
