#!/usr/bin/env python3
"""Verifier metrics for Seldinger synthetic DSA manifests.

This script turns typed generator artifacts into benchmark-verifier outputs. The
initial `identity` baseline deliberately uses the manifest ground truth as the
prediction so the harness can be smoke-tested before real models are plugged in.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from PIL import Image

PHASE_ORDER = ["precontrast", "arrival", "arterial_peak", "washout"]
DEFAULT_THRESHOLDS = (2, 5, 10)


def load_binary_mask(path: Path) -> set[tuple[int, int]]:
    image = Image.open(path).convert("L")
    pixels = image.load()
    w, h = image.size
    return {(x, y) for y in range(h) for x in range(w) if pixels[x, y] > 0}


def binary_mask_metrics_from_pixels(
    truth: set[tuple[int, int]],
    pred: set[tuple[int, int]],
) -> dict[str, float | int]:
    intersection = len(truth & pred)
    union = len(truth | pred)
    truth_area = len(truth)
    pred_area = len(pred)
    denom = truth_area + pred_area
    return {
        "truth_area_px": truth_area,
        "pred_area_px": pred_area,
        "intersection_px": intersection,
        "union_px": union,
        "iou": 1.0 if union == 0 else intersection / union,
        "dice": 1.0 if denom == 0 else (2 * intersection) / denom,
    }


def binary_mask_metrics(truth_path: Path, pred_path: Path) -> dict[str, float | int]:
    return binary_mask_metrics_from_pixels(load_binary_mask(truth_path), load_binary_mask(pred_path))


def threshold_mask_pixels(frame_path: Path, threshold: int = 180) -> set[tuple[int, int]]:
    image = Image.open(frame_path).convert("L")
    pixels = image.load()
    w, h = image.size
    return {(x, y) for y in range(h) for x in range(w) if pixels[x, y] >= threshold}


def brightest_pixel_xy(frame_path: Path) -> list[float]:
    image = Image.open(frame_path).convert("L")
    pixels = image.load()
    w, h = image.size
    best = (0, 0, -1)
    for y in range(h):
        for x in range(w):
            val = pixels[x, y]
            if val > best[2]:
                best = (x, y, val)
    return [float(best[0]), float(best[1])]


def mean_intensity(frame_path: Path) -> float:
    image = Image.open(frame_path).convert("L")
    hist = image.histogram()
    total = sum(hist)
    if total == 0:
        return 0.0
    return sum(value * count for value, count in enumerate(hist)) / total


def phases_from_frame_intensity(frame_paths: list[Path]) -> list[str]:
    if not frame_paths:
        return []
    values = [mean_intensity(path) for path in frame_paths]
    ranked = sorted(range(len(values)), key=lambda idx: values[idx])
    pred = ["arrival"] * len(values)
    pred[ranked[0]] = "precontrast"
    pred[ranked[-1]] = "arterial_peak"
    if len(values) > 2:
        pred[ranked[-2]] = "washout"
    return pred


def tip_localization_metrics(
    truth_xy: list[float] | tuple[float, float] | None,
    pred_xy: list[float] | tuple[float, float] | None,
    thresholds: Iterable[int] = DEFAULT_THRESHOLDS,
) -> dict[str, float | bool | None]:
    if truth_xy is None or pred_xy is None:
        result: dict[str, float | bool | None] = {"tip_error_px": None}
        for threshold in thresholds:
            result[f"within_{threshold}px"] = False
        return result
    dx = float(pred_xy[0]) - float(truth_xy[0])
    dy = float(pred_xy[1]) - float(truth_xy[1])
    error = math.hypot(dx, dy)
    result = {"tip_error_px": error}
    for threshold in thresholds:
        result[f"within_{threshold}px"] = error <= threshold
    return result


def bolus_phase_metrics(truth_phases: list[str], pred_phases: list[str]) -> dict[str, Any]:
    if len(truth_phases) != len(pred_phases):
        raise ValueError("truth_phases and pred_phases must have equal length")
    if not truth_phases:
        return {"phase_accuracy": 0.0, "phase_mae_frames": None, "phase_confusion": {}}
    order = {phase: idx for idx, phase in enumerate(PHASE_ORDER)}
    correct = 0
    abs_errors = []
    confusion: Counter[str] = Counter()
    for truth, pred in zip(truth_phases, pred_phases):
        if truth == pred:
            correct += 1
        confusion[f"{truth}->{pred}"] += 1
        abs_errors.append(abs(order.get(truth, 0) - order.get(pred, 0)))
    return {
        "phase_accuracy": correct / len(truth_phases),
        "phase_mae_frames": mean(abs_errors),
        "phase_confusion": dict(sorted(confusion.items())),
    }


def resolve_uri(root: Path, uri: str) -> Path:
    path = Path(uri)
    if path.is_absolute():
        return path
    return root / path


def centroid(mask_path: Path) -> tuple[float, float] | None:
    pixels = load_binary_mask(mask_path)
    if not pixels:
        return None
    return (mean([p[0] for p in pixels]), mean([p[1] for p in pixels]))


def temporal_centroid_drift(mask_paths: list[Path]) -> float | None:
    centers = [centroid(p) for p in mask_paths]
    centers = [c for c in centers if c is not None]
    if len(centers) < 2:
        return None
    return mean(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(centers, centers[1:]))


def frame_paths(directory: Path, prefix: str, count: int) -> list[Path]:
    return [directory / f"{prefix}_{idx:03d}.png" for idx in range(count)]


def ensure_files_exist(paths: Iterable[Path], sequence_id: str) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{sequence_id}: missing artifact(s): {missing[:5]}")


def identity_predictions(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "mask_uri": record["vessel_mask_sequence"]["uri"],
        "tip_xy_by_frame": record["catheter_tip_state"]["tip_xy_by_frame"],
        "phase_by_frame": record["bolus_curve"]["phase_by_frame"],
    }


def frame_threshold_predictions(frame_paths_: list[Path], threshold: int = 180) -> dict[str, Any]:
    return {
        "mask_pixels_by_frame": [threshold_mask_pixels(path, threshold=threshold) for path in frame_paths_],
        "tip_xy_by_frame": [brightest_pixel_xy(path) for path in frame_paths_],
        "phase_by_frame": phases_from_frame_intensity(frame_paths_),
        "params": {"mask_threshold": threshold, "tip_rule": "brightest_pixel", "phase_rule": "mean_intensity_rank"},
    }


def verify_record(record: dict[str, Any], root: Path, baseline: str = "identity") -> dict[str, Any]:
    sequence_id = record["sequence_id"]
    frame_count = int(record["dsa_frame_sequence"]["frame_count"])
    frame_dir = resolve_uri(root, record["dsa_frame_sequence"]["uri"])
    truth_mask_dir = resolve_uri(root, record["vessel_mask_sequence"]["uri"])

    frame_files = frame_paths(frame_dir, "frame", frame_count)
    truth_masks = frame_paths(truth_mask_dir, "mask", frame_count)
    ensure_files_exist(frame_files + truth_masks, sequence_id)

    if baseline == "identity":
        pred = identity_predictions(record)
        pred_mask_dir = resolve_uri(root, pred["mask_uri"])
        pred_masks = frame_paths(pred_mask_dir, "mask", frame_count)
        ensure_files_exist(pred_masks, sequence_id)
        pred_mask_pixels = [load_binary_mask(path) for path in pred_masks]
    elif baseline == "frame_threshold":
        pred = frame_threshold_predictions(frame_files)
        pred_mask_pixels = pred["mask_pixels_by_frame"]
    else:
        raise ValueError("baseline must be identity or frame_threshold")

    per_frame = []
    ious: list[float] = []
    dices: list[float] = []
    areas: list[int] = []
    tip_errors: list[float] = []
    threshold_hits: dict[int, list[bool]] = {t: [] for t in DEFAULT_THRESHOLDS}

    truth_tips = record["catheter_tip_state"]["tip_xy_by_frame"]
    pred_tips = pred["tip_xy_by_frame"]
    occlusion_flags = record.get("catheter_path", {}).get("occlusion_flags_by_frame", [False] * frame_count)

    for idx in range(frame_count):
        mask_metrics = binary_mask_metrics_from_pixels(load_binary_mask(truth_masks[idx]), pred_mask_pixels[idx])
        tip_metrics = tip_localization_metrics(truth_tips[idx], pred_tips[idx])
        ious.append(float(mask_metrics["iou"]))
        dices.append(float(mask_metrics["dice"]))
        areas.append(int(mask_metrics["truth_area_px"]))
        if tip_metrics["tip_error_px"] is not None:
            tip_errors.append(float(tip_metrics["tip_error_px"]))
        for threshold in DEFAULT_THRESHOLDS:
            threshold_hits[threshold].append(bool(tip_metrics[f"within_{threshold}px"]))
        per_frame.append({
            "frame_index": idx,
            "occluded": bool(occlusion_flags[idx]) if idx < len(occlusion_flags) else False,
            **mask_metrics,
            **tip_metrics,
        })

    phase = bolus_phase_metrics(record["bolus_curve"]["phase_by_frame"], pred["phase_by_frame"])
    non_empty_mask_rate = sum(1 for area in areas if area > 0) / frame_count if frame_count else 0.0
    mask_area_delta_abs_mean = mean(abs(b - a) for a, b in zip(areas, areas[1:])) if len(areas) > 1 else 0.0

    summary = {
        "sequence_id": sequence_id,
        "baseline": baseline,
        "baseline_params": pred.get("params", {}),
        "frame_count": frame_count,
        "failure_modes": record.get("failure_modes", []),
        "view": record.get("projection_view", {}).get("view"),
        "overlap_score": record.get("projection_view", {}).get("overlap_score"),
        "mean_iou": mean(ious) if ious else 0.0,
        "min_iou": min(ious) if ious else 0.0,
        "mean_dice": mean(dices) if dices else 0.0,
        "min_dice": min(dices) if dices else 0.0,
        "non_empty_mask_rate": non_empty_mask_rate,
        "mean_mask_area_px": mean(areas) if areas else 0.0,
        "mask_area_delta_abs_mean_px": mask_area_delta_abs_mean,
        "temporal_mask_centroid_drift_px": temporal_centroid_drift(truth_masks),
        "mean_tip_error_px": mean(tip_errors) if tip_errors else None,
        "max_tip_error_px": max(tip_errors) if tip_errors else None,
        "tip_occlusion_rate": sum(1 for v in occlusion_flags if v) / frame_count if frame_count else 0.0,
        "phase_accuracy": phase["phase_accuracy"],
        "phase_mae_frames": phase["phase_mae_frames"],
        "phase_confusion": phase["phase_confusion"],
        "per_frame": per_frame,
    }
    for threshold, hits in threshold_hits.items():
        summary[f"tip_within_{threshold}px_rate"] = sum(hits) / len(hits) if hits else 0.0
    return summary


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate(summaries: list[dict[str, Any]], manifest: Path, baseline: str) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        vals = [s[key] for s in summaries if s.get(key) is not None]
        return mean(vals) if vals else None

    failure_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        modes = summary.get("failure_modes") or ["none"]
        for mode in modes:
            failure_groups[mode].append(summary)

    by_failure_mode = {}
    for mode, rows in failure_groups.items():
        by_failure_mode[mode] = {
            "sequence_count": len(rows),
            "mean_iou": mean([r["mean_iou"] for r in rows]),
            "mean_tip_error_px": mean([r["mean_tip_error_px"] for r in rows if r["mean_tip_error_px"] is not None]),
            "phase_accuracy": mean([r["phase_accuracy"] for r in rows]),
        }

    return {
        "manifest": str(manifest),
        "baseline": baseline,
        "sequence_count": len(summaries),
        "aggregate": {
            "mean_iou": avg("mean_iou"),
            "min_iou": min([s["min_iou"] for s in summaries], default=None),
            "mean_dice": avg("mean_dice"),
            "mean_tip_error_px": avg("mean_tip_error_px"),
            "max_tip_error_px": max([s["max_tip_error_px"] for s in summaries if s["max_tip_error_px"] is not None], default=None),
            "tip_within_2px_rate": avg("tip_within_2px_rate"),
            "tip_within_5px_rate": avg("tip_within_5px_rate"),
            "tip_within_10px_rate": avg("tip_within_10px_rate"),
            "phase_accuracy": avg("phase_accuracy"),
            "phase_mae_frames": avg("phase_mae_frames"),
            "non_empty_mask_rate": avg("non_empty_mask_rate"),
            "tip_occlusion_rate": avg("tip_occlusion_rate"),
            "temporal_mask_centroid_drift_px": avg("temporal_mask_centroid_drift_px"),
        },
        "by_failure_mode": by_failure_mode,
        "sequences": summaries,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    agg = report["aggregate"]
    lines = [
        "# Synthetic DSA verifier report",
        "",
        f"Manifest: `{report['manifest']}`",
        f"Baseline: `{report['baseline']}`",
        f"Sequences: {report['sequence_count']}",
        "",
        "## Aggregate metrics",
        "",
    ]
    for key, value in agg.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Failure-mode slices", ""])
    for mode, row in sorted(report["by_failure_mode"].items()):
        lines.append(f"- {mode}")
        for key, value in row.items():
            if isinstance(value, float):
                lines.append(f"  - {key}: {value:.4f}")
            else:
                lines.append(f"  - {key}: {value}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Identity/oracle baseline should score perfectly on task metrics. Non-task sanity metrics such as mask area changes and centroid drift are descriptive checks for generator behavior, not model quality.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_jsonl", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--baseline", default="identity", choices=["identity", "frame_threshold"])
    ap.add_argument("--out-json", type=Path, default=Path("research/synthetic_dsa/outputs/reports/verifier_report.json"))
    ap.add_argument("--out-md", type=Path, default=Path("research/synthetic_dsa/outputs/reports/verifier_report.md"))
    args = ap.parse_args()

    records = load_manifest(args.manifest_jsonl)
    summaries = [verify_record(record, args.root, baseline=args.baseline) for record in records]
    report = aggregate(summaries, args.manifest_jsonl, args.baseline)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, args.out_md)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "sequence_count": len(summaries), "aggregate": report["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
