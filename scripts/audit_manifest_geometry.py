#!/usr/bin/env python3
"""Geometry and rendering QA audit for synthetic DSA manifests."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from PIL import Image, ImageDraw

TIP_DEVICE_PASS_PX = 2.5
TIP_PATH_PASS_PX = 4.0
MIN_DEVICE_PRESENCE_RATE = 0.95
MIN_VISIBLE_TIP_ON_DEVICE_RATE = 0.90


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_uri(root: Path, uri: str) -> Path:
    p = Path(uri)
    return p if p.is_absolute() else root / p


def frame_paths(directory: Path, prefix: str, count: int) -> list[Path]:
    return [directory / f"{prefix}_{idx:03d}.png" for idx in range(count)]


def mask_pixels(path: Path) -> set[tuple[int, int]]:
    image = Image.open(path).convert("L")
    px = image.load()
    w, h = image.size
    return {(x, y) for y in range(h) for x in range(w) if px[x, y] > 0}


def nearest_distance(point: list[float] | tuple[float, float] | None, pixels: Iterable[tuple[int, int]]) -> float | None:
    if point is None:
        return None
    pts = list(pixels)
    if not pts:
        return None
    x, y = float(point[0]), float(point[1])
    return min(math.hypot(px - x, py - y) for px, py in pts)


def point_to_polyline_distance(point: list[float] | tuple[float, float] | None, polyline: list[list[float]]) -> float | None:
    if point is None or len(polyline) < 2:
        return None
    px, py = float(point[0]), float(point[1])
    best = float("inf")
    for a, b in zip(polyline, polyline[1:]):
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        t = 0.0 if denom == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
        cx, cy = ax + t * dx, ay + t * dy
        best = min(best, math.hypot(px - cx, py - cy))
    return best


def infer_device_pixels(frame_path: Path, tip_xy: list[float] | None = None) -> set[tuple[int, int]]:
    """Fallback for old manifests without device masks: bright pixels near a tip/path."""
    image = Image.open(frame_path).convert("L")
    px = image.load()
    w, h = image.size
    if tip_xy is None:
        return {(x, y) for y in range(h) for x in range(w) if px[x, y] >= 225}
    tx, ty = float(tip_xy[0]), float(tip_xy[1])
    return {
        (x, y)
        for y in range(h)
        for x in range(w)
        if px[x, y] >= 180 and math.hypot(x - tx, y - ty) <= 8.0
    }


def audit_record(record: dict[str, Any], root: Path) -> dict[str, Any]:
    sequence_id = record["sequence_id"]
    frame_count = int(record["dsa_frame_sequence"]["frame_count"])
    frame_dir = resolve_uri(root, record["dsa_frame_sequence"]["uri"])
    vessel_dir = resolve_uri(root, record["vessel_mask_sequence"]["uri"])
    has_device_masks = "device_mask_sequence" in record
    device_dir = resolve_uri(root, record["device_mask_sequence"]["uri"]) if has_device_masks else None
    frames = frame_paths(frame_dir, "frame", frame_count)
    vessels = frame_paths(vessel_dir, "mask", frame_count)
    devices = frame_paths(device_dir, "device", frame_count) if device_dir else []

    tips = record.get("catheter_tip_state", {}).get("tip_xy_by_frame", [])
    visibility = record.get("catheter_tip_state", {}).get("visibility_by_frame", ["visible"] * frame_count)
    polyline = record.get("catheter_path", {}).get("polyline_px", [])
    per_frame = []
    tip_to_device_visible: list[float] = []
    tip_to_path: list[float] = []
    device_present = []
    frame_failures: list[str] = []

    for idx in range(frame_count):
        tip = tips[idx] if idx < len(tips) else None
        vessel_px = mask_pixels(vessels[idx])
        if has_device_masks:
            device_px = mask_pixels(devices[idx])
        else:
            device_px = infer_device_pixels(frames[idx], tip)
        device_present.append(bool(device_px))
        d_device = nearest_distance(tip, device_px)
        d_vessel = nearest_distance(tip, vessel_px)
        d_path = point_to_polyline_distance(tip, polyline)
        is_visible = visibility[idx] == "visible" if idx < len(visibility) else True
        if d_path is not None:
            tip_to_path.append(d_path)
        if is_visible and d_device is not None:
            tip_to_device_visible.append(d_device)
        if is_visible and (d_device is None or d_device > TIP_DEVICE_PASS_PX):
            frame_failures.append("tip_not_on_device")
        if d_path is not None and d_path > TIP_PATH_PASS_PX:
            frame_failures.append("tip_far_from_path")
        per_frame.append({
            "frame_index": idx,
            "visible": bool(is_visible),
            "device_pixels": len(device_px),
            "vessel_pixels": len(vessel_px),
            "tip_to_device_px": None if d_device is None else round(d_device, 4),
            "tip_to_vessel_px": None if d_vessel is None else round(d_vessel, 4),
            "tip_to_path_px": None if d_path is None else round(d_path, 4),
        })

    device_presence_rate = sum(device_present) / frame_count if frame_count else 0.0
    visible_tip_on_device_rate = sum(1 for d in tip_to_device_visible if d <= TIP_DEVICE_PASS_PX) / len(tip_to_device_visible) if tip_to_device_visible else 0.0
    mean_tip_to_device = mean(tip_to_device_visible) if tip_to_device_visible else None
    mean_tip_to_path = mean(tip_to_path) if tip_to_path else None
    failures = set(frame_failures)
    if not has_device_masks:
        failures.add("missing_device_mask_sequence")
    if device_presence_rate < MIN_DEVICE_PRESENCE_RATE:
        failures.add("device_missing_frames")
    if visible_tip_on_device_rate < MIN_VISIBLE_TIP_ON_DEVICE_RATE:
        failures.add("tip_not_on_device")
    if mean_tip_to_path is not None and mean_tip_to_path > TIP_PATH_PASS_PX:
        failures.add("tip_far_from_path")
    return {
        "sequence_id": sequence_id,
        "passes_qa": not failures,
        "has_device_mask_sequence": has_device_masks,
        "device_presence_rate": device_presence_rate,
        "visible_tip_on_device_rate": visible_tip_on_device_rate,
        "mean_visible_tip_to_device_px": mean_tip_to_device,
        "mean_tip_to_path_px": mean_tip_to_path,
        "qa_failures": sorted(failures),
        "frames": per_frame,
    }


def paint_mask(base: Image.Image, pixels: set[tuple[int, int]], color: tuple[int, int, int], alpha: float) -> None:
    out = base.load()
    for x, y in pixels:
        if 0 <= x < base.width and 0 <= y < base.height:
            r, g, b = out[x, y]
            out[x, y] = (int(r * (1 - alpha) + color[0] * alpha), int(g * (1 - alpha) + color[1] * alpha), int(b * (1 - alpha) + color[2] * alpha))


def draw_cross(draw: ImageDraw.ImageDraw, xy: list[float] | tuple[float, float], color: tuple[int, int, int], radius: int = 4) -> None:
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    draw.line((x - radius, y, x + radius, y), fill=color, width=1)
    draw.line((x, y - radius, x, y + radius), fill=color, width=1)


def render_diagnostic_overlay(record: dict[str, Any], root: Path, frame_index: int = 0) -> Image.Image:
    frame_count = int(record["dsa_frame_sequence"]["frame_count"])
    idx = max(0, min(frame_count - 1, frame_index))
    frame_dir = resolve_uri(root, record["dsa_frame_sequence"]["uri"])
    vessel_dir = resolve_uri(root, record["vessel_mask_sequence"]["uri"])
    frame = Image.open(frame_dir / f"frame_{idx:03d}.png").convert("L")
    base = Image.merge("RGB", (frame, frame, frame))
    vessel_px = mask_pixels(vessel_dir / f"mask_{idx:03d}.png")
    paint_mask(base, vessel_px, (0, 255, 60), alpha=0.55)
    if "device_mask_sequence" in record:
        device_dir = resolve_uri(root, record["device_mask_sequence"]["uri"])
        device_px = mask_pixels(device_dir / f"device_{idx:03d}.png")
    else:
        tips = record.get("catheter_tip_state", {}).get("tip_xy_by_frame", [])
        device_px = infer_device_pixels(frame_dir / f"frame_{idx:03d}.png", tips[idx] if idx < len(tips) else None)
    paint_mask(base, device_px, (40, 160, 255), alpha=0.45)
    draw = ImageDraw.Draw(base)
    polyline = record.get("catheter_path", {}).get("polyline_px", [])
    if len(polyline) >= 2:
        draw.line([(round(x), round(y)) for x, y in polyline], fill=(255, 215, 0), width=1)
    tips = record.get("catheter_tip_state", {}).get("tip_xy_by_frame", [])
    if idx < len(tips) and tips[idx] is not None:
        draw_cross(draw, tips[idx], (255, 40, 40), radius=max(3, base.width // 32))
    return base


def write_diagnostics(records: list[dict[str, Any]], results: list[dict[str, Any]], root: Path, out_dir: Path, limit: int = 12) -> list[str]:
    by_id = {r["sequence_id"]: r for r in records}
    chosen = sorted(results, key=lambda r: (r["passes_qa"], -(r.get("mean_visible_tip_to_device_px") or 0)))[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for res in chosen:
        rec = by_id[res["sequence_id"]]
        worst_idx = 0
        if res["frames"]:
            worst = max(res["frames"], key=lambda f: f.get("tip_to_device_px") or -1)
            worst_idx = int(worst["frame_index"])
        img = render_diagnostic_overlay(rec, root, frame_index=worst_idx)
        out = out_dir / f"{res['sequence_id']}_qa_overlay.png"
        img.save(out)
        written.append(str(out))
    return written


def aggregate(results: list[dict[str, Any]], manifest: Path) -> dict[str, Any]:
    failures: dict[str, int] = {}
    for r in results:
        for f in r["qa_failures"]:
            failures[f] = failures.get(f, 0) + 1
    return {
        "manifest": str(manifest),
        "sequence_count": len(results),
        "pass_count": sum(1 for r in results if r["passes_qa"]),
        "pass_rate": sum(1 for r in results if r["passes_qa"]) / len(results) if results else 0.0,
        "failure_counts": dict(sorted(failures.items())),
        "aggregate": {
            "device_presence_rate_mean": mean([r["device_presence_rate"] for r in results]) if results else 0.0,
            "visible_tip_on_device_rate_mean": mean([r["visible_tip_on_device_rate"] for r in results]) if results else 0.0,
            "mean_visible_tip_to_device_px": mean([r["mean_visible_tip_to_device_px"] for r in results if r["mean_visible_tip_to_device_px"] is not None]) if any(r["mean_visible_tip_to_device_px"] is not None for r in results) else None,
            "mean_tip_to_path_px": mean([r["mean_tip_to_path_px"] for r in results if r["mean_tip_to_path_px"] is not None]) if any(r["mean_tip_to_path_px"] is not None for r in results) else None,
        },
        "sequences": results,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Synthetic DSA geometry QA audit",
        "",
        f"Manifest: `{report['manifest']}`",
        f"Sequences: {report['sequence_count']}",
        f"Pass rate: {report['pass_rate']:.4f}",
        "",
        "## Aggregate",
        "",
    ]
    for k, v in report["aggregate"].items():
        lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    lines.extend(["", "## Failure counts", ""])
    for k, v in report["failure_counts"].items():
        lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_jsonl", type=Path)
    ap.add_argument("--root", type=Path, default=Path("research/synthetic_dsa"))
    ap.add_argument("--out-json", type=Path, default=Path("research/synthetic_dsa/outputs/reports/geometry_qa_report.json"))
    ap.add_argument("--out-md", type=Path, default=Path("research/synthetic_dsa/outputs/reports/geometry_qa_report.md"))
    ap.add_argument("--diagnostics-dir", type=Path, default=Path("research/synthetic_dsa/outputs/figures/geometry_qa"))
    ap.add_argument("--diagnostic-limit", type=int, default=12)
    args = ap.parse_args()
    records = load_manifest(args.manifest_jsonl)
    results = [audit_record(record, args.root) for record in records]
    report = aggregate(results, args.manifest_jsonl)
    report["diagnostic_overlays"] = write_diagnostics(records, results, args.root, args.diagnostics_dir, limit=args.diagnostic_limit)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, args.out_md)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "pass_rate": report["pass_rate"], "failure_counts": report["failure_counts"], "diagnostic_overlays": report["diagnostic_overlays"][:3]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
