#!/usr/bin/env python3
"""Create overlay and failure-case figures for synthetic DSA verifier reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def sequence_score(summary: dict[str, Any], metric: str = "composite") -> float:
    iou = float(summary.get("mean_iou") or 0.0)
    tip = float(summary.get("mean_tip_error_px") or 0.0)
    phase_acc = float(summary.get("phase_accuracy") or 0.0)
    if metric == "mean_iou":
        return 1.0 - iou
    if metric == "tip_error":
        return tip
    if metric == "phase_error":
        return 1.0 - phase_acc
    if metric != "composite":
        raise ValueError("metric must be composite, mean_iou, tip_error, or phase_error")
    return (2.0 * (1.0 - iou)) + min(tip / 10.0, 2.0) + (0.5 * (1.0 - phase_acc))


def select_worst_sequences(report: dict[str, Any], metric: str = "composite", limit: int = 6) -> list[dict[str, Any]]:
    sequences = list(report.get("sequences", []))
    return sorted(sequences, key=lambda row: sequence_score(row, metric=metric), reverse=True)[:limit]


def frame_score(frame: dict[str, Any]) -> float:
    iou = float(frame.get("iou") or 0.0)
    tip = float(frame.get("tip_error_px") or 0.0)
    return (2.0 * (1.0 - iou)) + min(tip / 10.0, 2.0)


def select_worst_frame(summary: dict[str, Any]) -> int:
    frames = list(summary.get("per_frame") or [])
    if not frames:
        return 0
    worst = max(frames, key=frame_score)
    return int(worst.get("frame_index", 0))


def draw_cross(draw: ImageDraw.ImageDraw, xy: list[float] | tuple[float, float], color: tuple[int, int, int], radius: int = 4) -> None:
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    draw.line((x - radius, y, x + radius, y), fill=color, width=1)
    draw.line((x, y - radius, x, y + radius), fill=color, width=1)


def overlay_frame(
    frame_path: Path,
    mask_path: Path,
    tip_xy: list[float] | tuple[float, float] | None,
    predicted_tip_xy: list[float] | tuple[float, float] | None = None,
    label: str = "",
) -> Image.Image:
    frame = Image.open(frame_path).convert("L")
    mask = Image.open(mask_path).convert("L").resize(frame.size)
    base = Image.merge("RGB", (frame, frame, frame))
    pix = base.load()
    mp = mask.load()
    w, h = base.size
    for y in range(h):
        for x in range(w):
            if mp[x, y] > 0:
                r, g, b = pix[x, y]
                pix[x, y] = (int(r * 0.35), min(255, int(g * 0.35) + 170), int(b * 0.35))
    draw = ImageDraw.Draw(base)
    if label:
        draw.rectangle((0, 0, min(w, 6 * len(label) + 8), 13), fill=(0, 0, 0))
        draw.text((3, 2), label, fill=(255, 255, 255))
    radius = max(3, min(w, h) // 32)
    if tip_xy is not None and predicted_tip_xy is not None:
        draw.line((float(tip_xy[0]), float(tip_xy[1]), float(predicted_tip_xy[0]), float(predicted_tip_xy[1])), fill=(255, 220, 40), width=1)
    if predicted_tip_xy is not None:
        draw_cross(draw, predicted_tip_xy, (40, 240, 255), radius=radius)
    if tip_xy is not None:
        draw_cross(draw, tip_xy, (255, 40, 40), radius=radius)
    return base


def make_panel(record: dict[str, Any], summary: dict[str, Any], root: Path, thumb_size: tuple[int, int]) -> Image.Image:
    seq_id = record["sequence_id"]
    frame_idx = select_worst_frame(summary)
    frame_dir = resolve_uri(root, record["dsa_frame_sequence"]["uri"])
    mask_dir = resolve_uri(root, record["vessel_mask_sequence"]["uri"])
    frame_path = frame_dir / f"frame_{frame_idx:03d}.png"
    mask_path = mask_dir / f"mask_{frame_idx:03d}.png"
    tips = record.get("catheter_tip_state", {}).get("tip_xy_by_frame", [])
    tip_xy = tips[frame_idx] if frame_idx < len(tips) else None
    frame_summary = next((f for f in summary.get("per_frame", []) if int(f.get("frame_index", -1)) == frame_idx), {})
    predicted_tip_xy = frame_summary.get("predicted_tip_xy")
    tip_error = frame_summary.get("tip_error_px", summary.get("mean_tip_error_px"))
    label = f"{seq_id} f{frame_idx}"
    overlay = overlay_frame(frame_path, mask_path, tip_xy, predicted_tip_xy=predicted_tip_xy, label=label)
    overlay.thumbnail(thumb_size)
    panel_h = thumb_size[1] + 90
    panel = Image.new("RGB", (thumb_size[0], panel_h), (8, 8, 8))
    panel.paste(overlay, (0, 0))
    draw = ImageDraw.Draw(panel)
    modes = ", ".join(summary.get("failure_modes") or ["none"])
    lines = [
        "truth=red pred=cyan error-line=yellow",
        f"mean_iou={float(summary.get('mean_iou') or 0):.3f} tip={float(summary.get('mean_tip_error_px') or 0):.2f}px f_tip={float(tip_error or 0):.2f}px",
        f"phase_acc={float(summary.get('phase_accuracy') or 0):.3f} view={summary.get('view')}",
        modes[:70],
    ]
    y = thumb_size[1] + 6
    for line in lines:
        draw.text((4, y), line, fill=(230, 230, 230))
        y += 16
    return panel


def make_failure_sheet(
    manifest_path: Path,
    report_path: Path,
    root: Path,
    out_path: Path,
    limit: int = 6,
    metric: str = "composite",
    cols: int = 2,
) -> Path:
    records = {r["sequence_id"]: r for r in load_manifest(manifest_path)}
    report = load_json(report_path)
    selected = select_worst_sequences(report, metric=metric, limit=limit)
    if not selected:
        raise ValueError("report contains no sequences")
    panels = []
    for summary in selected:
        record = records.get(summary["sequence_id"])
        if record is None:
            continue
        panels.append(make_panel(record, summary, root=root, thumb_size=(256, 256)))
    if not panels:
        raise ValueError("no selected sequences matched manifest records")
    cols = max(1, cols)
    rows = (len(panels) + cols - 1) // cols
    panel_w = max(p.width for p in panels)
    panel_h = max(p.height for p in panels)
    sheet = Image.new("RGB", (panel_w * cols, panel_h * rows), (0, 0, 0))
    for idx, panel in enumerate(panels):
        x = (idx % cols) * panel_w
        y = (idx // cols) * panel_h
        sheet.paste(panel, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_jsonl", type=Path)
    ap.add_argument("report_json", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("research/synthetic_dsa/outputs/figures/failure_cases.png"))
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--metric", choices=["composite", "mean_iou", "tip_error", "phase_error"], default="composite")
    ap.add_argument("--cols", type=int, default=2)
    args = ap.parse_args()
    out = make_failure_sheet(args.manifest_jsonl, args.report_json, args.root, args.out, limit=args.limit, metric=args.metric, cols=args.cols)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
