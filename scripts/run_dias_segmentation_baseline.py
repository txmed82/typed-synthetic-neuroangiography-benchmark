#!/usr/bin/env python3
"""Dependency-light DIAS vessel segmentation baseline.

The baseline learns a single grayscale threshold and projection feature from
labeled training sequences. Candidate features are max intensity, min intensity,
mean intensity, and temporal range over frames. DIAS vessels are often best
separated by temporal range because contrast changes over the sequence. This is
intentionally simple: it creates an external DSA sanity check before any
GPU/RunPod spend.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageFilter


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_mask_pixels(path: Path) -> set[tuple[int, int]]:
    image = Image.open(path).convert("L")
    px = image.load()
    w, h = image.size
    return {(x, y) for y in range(h) for x in range(w) if px[x, y] > 0}


def projection_image(frame_paths: list[Path], mode: str = "range") -> Image.Image:
    if not frame_paths:
        raise ValueError("frame_paths cannot be empty")
    if mode not in {"max", "min", "mean", "range"}:
        raise ValueError("mode must be one of max, min, mean, range")
    images = [Image.open(path).convert("L") for path in frame_paths]
    w, h = images[0].size
    out = Image.new("L", (w, h), 0)
    out_px = out.load()
    loads = [img.load() for img in images]
    for y in range(h):
        for x in range(w):
            values = [px[x, y] for px in loads]
            if mode == "max":
                out_px[x, y] = max(values)
            elif mode == "min":
                out_px[x, y] = min(values)
            elif mode == "mean":
                out_px[x, y] = int(round(sum(values) / len(values)))
            else:
                out_px[x, y] = max(values) - min(values)
    for img in images:
        img.close()
    return out


def threshold_mask_image(image: Image.Image, threshold: int, polarity: str = ">=") -> Image.Image:
    """Return a 0/255 binary image from a grayscale threshold."""
    if polarity not in {">=", "<="}:
        raise ValueError("polarity must be >= or <=")
    image = image.convert("L")
    if polarity == ">=":
        return image.point(lambda value: 255 if value >= threshold else 0, mode="L")
    return image.point(lambda value: 255 if value <= threshold else 0, mode="L")


def threshold_pixels(image: Image.Image, threshold: int, polarity: str = ">=") -> set[tuple[int, int]]:
    mask = threshold_mask_image(image, threshold, polarity=polarity)
    px = mask.load()
    w, h = mask.size
    return {(x, y) for y in range(h) for x in range(w) if px[x, y] > 0}


def mask_pixels_from_image(image: Image.Image) -> set[tuple[int, int]]:
    mask = image.convert("L")
    px = mask.load()
    w, h = mask.size
    return {(x, y) for y in range(h) for x in range(w) if px[x, y] > 0}


def binary_metrics(truth: set[tuple[int, int]], pred: set[tuple[int, int]]) -> dict[str, float | int]:
    inter = len(truth & pred)
    union = len(truth | pred)
    denom = len(truth) + len(pred)
    return {
        "truth_area_px": len(truth),
        "pred_area_px": len(pred),
        "intersection_px": inter,
        "union_px": union,
        "iou": 1.0 if union == 0 else inter / union,
        "dice": 1.0 if denom == 0 else (2 * inter) / denom,
    }


def binary_image_metrics(truth_image: Image.Image, pred_image: Image.Image) -> dict[str, float | int]:
    """Compute binary metrics by streaming bytes; avoids large pixel sets for DIAS-sized images."""
    truth_bytes = truth_image.convert("L").tobytes()
    pred_bytes = pred_image.convert("L").tobytes()
    truth_area = pred_area = inter = union = 0
    for truth_value, pred_value in zip(truth_bytes, pred_bytes):
        truth = truth_value > 0
        pred = pred_value > 0
        truth_area += int(truth)
        pred_area += int(pred)
        inter += int(truth and pred)
        union += int(truth or pred)
    denom = truth_area + pred_area
    return {
        "truth_area_px": truth_area,
        "pred_area_px": pred_area,
        "intersection_px": inter,
        "union_px": union,
        "iou": 1.0 if union == 0 else inter / union,
        "dice": 1.0 if denom == 0 else (2 * inter) / denom,
    }


def morphology(mask: Image.Image, open_radius: int = 0, close_radius: int = 0) -> Image.Image:
    """Binary morphology using PIL min/max filters: open removes speckles, close bridges gaps."""
    out = mask.convert("L")
    if close_radius > 0:
        size = 2 * close_radius + 1
        out = out.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    if open_radius > 0:
        size = 2 * open_radius + 1
        out = out.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size))
    return out.point(lambda value: 255 if value > 0 else 0, mode="L")


def remove_small_components(mask: Image.Image, min_area: int = 0) -> Image.Image:
    """Remove connected components smaller than min_area with dependency-free 8-neighborhood BFS."""
    if min_area <= 1:
        return mask.convert("L").point(lambda value: 255 if value > 0 else 0, mode="L")
    binary = mask.convert("L")
    w, h = binary.size
    data = bytearray(1 if value > 0 else 0 for value in binary.tobytes())
    visited = bytearray(w * h)
    keep = bytearray(w * h)
    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    for start_idx, value in enumerate(data):
        if not value or visited[start_idx]:
            continue
        component: list[int] = []
        queue: deque[int] = deque([start_idx])
        visited[start_idx] = 1
        while queue:
            idx = queue.popleft()
            component.append(idx)
            x = idx % w
            y = idx // w
            for dx, dy in neighbors:
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                nidx = ny * w + nx
                if data[nidx] and not visited[nidx]:
                    visited[nidx] = 1
                    queue.append(nidx)
        if len(component) >= min_area:
            for idx in component:
                keep[idx] = 255
    return Image.frombytes("L", (w, h), bytes(keep))


def apply_postprocess(mask: Image.Image, model: dict[str, Any]) -> Image.Image:
    out = morphology(mask, open_radius=int(model.get("open_radius", 0)), close_radius=int(model.get("close_radius", 0)))
    out = remove_small_components(out, min_area=int(model.get("min_component_area", 0)))
    return out


def resolve(root: Path, uri: str | None) -> Path | None:
    if uri is None:
        return None
    path = Path(uri)
    return path if path.is_absolute() else root / path


def record_frame_paths(record: dict[str, Any], root: Path) -> list[Path]:
    if "frame_files" in record:
        return [resolve(root, uri) for uri in record["frame_files"]]  # type: ignore[list-item]
    frame_dir = resolve(root, record["dsa_frame_sequence"]["uri"])
    count = int(record["dsa_frame_sequence"]["frame_count"])
    return [frame_dir / f"frame_{i:03d}.png" for i in range(count)]  # type: ignore[operator]


def label_path(record: dict[str, Any], root: Path) -> Path | None:
    return resolve(root, record.get("vessel_mask_sequence", {}).get("uri"))


def threshold_score(records: list[dict[str, Any]], root: Path, threshold: int, mode: str, polarity: str) -> float:
    scores = []
    for record in records:
        lp = label_path(record, root)
        if lp is None or not lp.exists():
            continue
        mip = projection_image(record_frame_paths(record, root), mode=mode)
        truth_img = Image.open(lp).convert("L")
        pred_img = threshold_mask_image(mip, threshold, polarity=polarity)
        scores.append(float(binary_image_metrics(truth_img, pred_img)["dice"]))
        truth_img.close()
    return mean(scores) if scores else 0.0


def sampled_feature_truth(record: dict[str, Any], root: Path, mode: str, stride: int = 8) -> tuple[list[int], list[bool]]:
    """Return sampled projection values and truth labels for fast threshold search."""
    lp = label_path(record, root)
    if lp is None or not lp.exists():
        return [], []
    proj = projection_image(record_frame_paths(record, root), mode=mode)
    truth_img = Image.open(lp).convert("L")
    pp = proj.load()
    tp = truth_img.load()
    w, h = proj.size
    values: list[int] = []
    labels: list[bool] = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            values.append(int(pp[x, y]))
            labels.append(bool(tp[x, y] > 0))
    truth_img.close()
    return values, labels


def sampled_threshold_dice(samples: list[tuple[list[int], list[bool]]], threshold: int, polarity: str) -> float:
    dices = []
    for values, labels in samples:
        if not values:
            continue
        tp = fp = fn = 0
        for value, truth in zip(values, labels):
            pred = value >= threshold if polarity == ">=" else value <= threshold
            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif truth:
                fn += 1
        denom = 2 * tp + fp + fn
        dices.append(1.0 if denom == 0 else (2 * tp) / denom)
    return mean(dices) if dices else 0.0


def postprocess_score(records: list[dict[str, Any]], root: Path, model: dict[str, Any]) -> float:
    scores = []
    for record in records:
        lp = label_path(record, root)
        if lp is None or not lp.exists():
            continue
        proj = projection_image(record_frame_paths(record, root), mode=model["projection"])
        raw = threshold_mask_image(proj, int(model["threshold"]), polarity=model["polarity"])
        pred_img = apply_postprocess(raw, model)
        truth_img = Image.open(lp).convert("L")
        scores.append(float(binary_image_metrics(truth_img, pred_img)["dice"]))
        truth_img.close()
    return mean(scores) if scores else 0.0


def learn_postprocess(records: list[dict[str, Any]], root: Path, threshold_model: dict[str, Any]) -> dict[str, Any]:
    """Tune a small, conservative morphology grid on training Dice."""
    best = {**threshold_model, "open_radius": 0, "close_radius": 0, "min_component_area": 0, "postprocess_train_mean_dice": postprocess_score(records, root, threshold_model)}
    for open_radius in [0, 1]:
        for close_radius in [0, 1, 2]:
            for min_component_area in [0, 16, 64, 256]:
                candidate = {**threshold_model, "open_radius": open_radius, "close_radius": close_radius, "min_component_area": min_component_area}
                score = postprocess_score(records, root, candidate)
                if score > float(best["postprocess_train_mean_dice"]):
                    best = {**candidate, "postprocess_train_mean_dice": score}
    return best


def learn_model(records: list[dict[str, Any]], root: Path, candidates: range = range(1, 256, 4), baseline: str = "projection_threshold") -> dict[str, Any]:
    if baseline not in {"projection_threshold", "projection_morphology"}:
        raise ValueError("baseline must be projection_threshold or projection_morphology")
    if not records:
        base = {"projection": "range", "threshold": 20, "polarity": ">="}
        return {**base, "open_radius": 0, "close_radius": 0, "min_component_area": 0} if baseline == "projection_morphology" else base
    best = {"projection": "range", "threshold": 20, "polarity": ">="}
    best_score = -1.0
    for mode in ["range", "max", "min", "mean"]:
        samples = [sampled_feature_truth(record, root, mode=mode) for record in records]
        for polarity in ([">="] if mode == "range" else [">=", "<="]):
            for threshold in candidates:
                score = sampled_threshold_dice(samples, threshold, polarity=polarity)
                if score > best_score:
                    best_score = score
                    best = {"projection": mode, "threshold": threshold, "polarity": polarity, "train_mean_dice": score}
    if baseline == "projection_morphology":
        return learn_postprocess(records, root, best)
    return best


def evaluate_records(records: list[dict[str, Any]], root: Path, model: dict[str, Any], pred_dir: Path | None = None) -> list[dict[str, Any]]:
    rows = []
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        lp = label_path(record, root)
        if lp is None or not lp.exists():
            continue
        mip = projection_image(record_frame_paths(record, root), mode=model["projection"])
        raw_pred_img = threshold_mask_image(mip, int(model["threshold"]), polarity=model["polarity"])
        pred_img = apply_postprocess(raw_pred_img, model)
        truth_img = Image.open(lp).convert("L")
        metrics = binary_image_metrics(truth_img, pred_img)
        pred_uri = None
        if pred_dir:
            pred_path = pred_dir / f"{record['sequence_id']}_{model['projection']}_{model['name']}_mask.png"
            pred_img.save(pred_path)
            pred_uri = str(pred_path)
        row = {
            "sequence_id": record["sequence_id"],
            "split": record["split"],
            "frame_count": record["dsa_frame_sequence"]["frame_count"],
            "model_name": model["name"],
            "projection": model["projection"],
            "threshold": model["threshold"],
            "polarity": model["polarity"],
            "open_radius": model.get("open_radius", 0),
            "close_radius": model.get("close_radius", 0),
            "min_component_area": model.get("min_component_area", 0),
            "prediction_uri": pred_uri,
            **metrics,
        }
        rows.append(row)
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


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# DIAS vessel segmentation baseline",
        "",
        f"Dataset root: `{report['dataset_root']}`",
        f"Manifest: `{report['manifest']}`",
        f"Train split: `{report['train_split']}` ({report['train_sequence_count']} sequences)",
        f"Eval split: `{report['eval_split']}` ({report['eval_sequence_count']} sequences)",
        f"Model: `{report['model']['name']}` projection={report['model']['projection']} threshold={report['model']['threshold']} polarity={report['model']['polarity']} open={report['model'].get('open_radius', 0)} close={report['model'].get('close_radius', 0)} min_component_area={report['model'].get('min_component_area', 0)}",
        "",
        "## Aggregate",
        "",
    ]
    for k, v in report["aggregate"].items():
        lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    lines.extend([
        "",
        "## Readout",
        "",
        "- This is a deliberately weak projection+threshold baseline for external DSA sanity checks.",
        "- It validates DIAS manifest wiring and produces a first non-GPU vessel-segmentation reference point.",
        "- It does not evaluate catheter-tip/device labels because DIAS does not provide those annotations.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_baseline(
    manifest_path: Path,
    dataset_root: Path,
    train_split: str = "training",
    eval_split: str = "validation",
    pred_dir: Path | None = None,
    baseline: str = "projection_threshold",
) -> dict[str, Any]:
    records = load_manifest(manifest_path)
    labeled = [r for r in records if r.get("has_labels")]
    train = [r for r in labeled if r.get("split") == train_split]
    eval_records = [r for r in labeled if r.get("split") == eval_split]
    if not train:
        raise ValueError(f"no labeled train records for split {train_split}")
    if not eval_records:
        raise ValueError(f"no labeled eval records for split {eval_split}")
    learned = learn_model(train, dataset_root, baseline=baseline)
    model = {
        "name": baseline,
        "version": "0.3.0" if baseline == "projection_morphology" else "0.2.0",
        **learned,
        "threshold_search": "projection in range/max/min/mean; threshold 1..255 step 4 on train Dice",
    }
    if baseline == "projection_morphology":
        model["postprocess_search"] = "train Dice grid over close_radius in {0,1,2}, open_radius in {0,1}, min_component_area in {0,16,64,256}"
    rows = evaluate_records(eval_records, dataset_root, model, pred_dir=pred_dir)
    return {
        "dataset": "DIAS",
        "dataset_root": str(dataset_root),
        "manifest": str(manifest_path),
        "train_split": train_split,
        "eval_split": eval_split,
        "train_sequence_count": len(train),
        "eval_sequence_count": len(rows),
        "model": model,
        "aggregate": aggregate(rows),
        "per_sequence": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_jsonl", type=Path)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--train-split", default="training")
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--baseline", choices=["projection_threshold", "projection_morphology"], default="projection_threshold")
    ap.add_argument("--pred-dir", type=Path, default=Path("research/synthetic_dsa/outputs/dias_predictions/projection_threshold"))
    ap.add_argument("--out-json", type=Path, default=Path("research/synthetic_dsa/outputs/reports/dias_validation_projection_threshold_report.json"))
    ap.add_argument("--out-md", type=Path, default=Path("research/synthetic_dsa/outputs/reports/dias_validation_projection_threshold_report.md"))
    args = ap.parse_args()

    report = run_baseline(args.manifest_jsonl, args.dataset_root, train_split=args.train_split, eval_split=args.eval_split, pred_dir=args.pred_dir, baseline=args.baseline)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, args.out_md)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "aggregate": report["aggregate"], "model": report["model"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
