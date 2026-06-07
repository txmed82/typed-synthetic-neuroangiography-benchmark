#!/usr/bin/env python3
"""Create DIAS vessel-segmentation overlay comparison figures.

This produces paper-candidate contact sheets comparing a weak temporal
projection-threshold baseline with the projection-morphology baseline. It is
intentionally dependency-light (PIL only) so the DIAS external sanity figures can
be regenerated without GPU/cloud setup.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_dias_segmentation_baseline import label_path, projection_image, record_frame_paths  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_path(path: str | Path, cwd: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (cwd or Path.cwd()) / p


def row_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["sequence_id"]: row for row in report.get("per_sequence", [])}


def select_sequences(
    threshold_report: dict[str, Any],
    morphology_report: dict[str, Any],
    limit: int = 6,
) -> list[str]:
    """Select a mix of hard DIAS cases and morphology-vs-threshold deltas."""
    threshold = row_map(threshold_report)
    morph = row_map(morphology_report)
    common = sorted(set(threshold) & set(morph))
    hard = sorted(common, key=lambda sid: float(morph[sid].get("dice") or 0.0))[: max(1, limit // 2)]
    deltas = sorted(
        common,
        key=lambda sid: abs(float(morph[sid].get("dice") or 0.0) - float(threshold[sid].get("dice") or 0.0)),
        reverse=True,
    )
    selected: list[str] = []
    for sid in hard + deltas:
        if sid not in selected:
            selected.append(sid)
        if len(selected) >= limit:
            break
    return selected


def overlay_prediction(projection: Image.Image, truth_mask: Image.Image, pred_mask: Image.Image, label: str) -> Image.Image:
    """Overlay truth/prediction masks on a projection image.

    Color key:
    - green: ground-truth vessel only
    - red: prediction only
    - yellow: truth/prediction overlap
    """
    base_l = projection.convert("L")
    truth_l = truth_mask.convert("L").resize(base_l.size)
    pred_l = pred_mask.convert("L").resize(base_l.size)
    out = Image.merge("RGB", (base_l, base_l, base_l))
    pix = out.load()
    tp = truth_l.load()
    pp = pred_l.load()
    w, h = out.size
    for y in range(h):
        for x in range(w):
            truth = tp[x, y] > 0
            pred = pp[x, y] > 0
            if truth and pred:
                pix[x, y] = (245, 215, 35)
            elif truth:
                r, g, b = pix[x, y]
                pix[x, y] = (int(r * 0.25), min(255, int(g * 0.25) + 190), int(b * 0.25))
            elif pred:
                r, g, b = pix[x, y]
                pix[x, y] = (min(255, int(r * 0.25) + 210), int(g * 0.25), int(b * 0.25))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, min(w, 7 * len(label) + 10), 16), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255))
    return out


def make_sequence_panel(
    record: dict[str, Any],
    threshold_row: dict[str, Any],
    morphology_row: dict[str, Any],
    dataset_root: Path,
    thumb_size: tuple[int, int] = (280, 280),
) -> Image.Image:
    mode = str(morphology_row.get("projection") or threshold_row.get("projection") or "range")
    projection = projection_image(record_frame_paths(record, dataset_root), mode=mode)
    truth_path = label_path(record, dataset_root)
    if truth_path is None:
        raise ValueError(f"missing truth label for {record['sequence_id']}")
    truth = Image.open(truth_path).convert("L")
    threshold_pred = Image.open(resolve_path(threshold_row["prediction_uri"])).convert("L")
    morphology_pred = Image.open(resolve_path(morphology_row["prediction_uri"])).convert("L")

    sid = record["sequence_id"]
    left = overlay_prediction(
        projection,
        truth,
        threshold_pred,
        f"{sid} threshold Dice={float(threshold_row['dice']):.3f}",
    )
    right = overlay_prediction(
        projection,
        truth,
        morphology_pred,
        f"{sid} morph Dice={float(morphology_row['dice']):.3f}",
    )
    left.thumbnail(thumb_size)
    right.thumbnail(thumb_size)

    text_h = 68
    panel_w = (thumb_size[0] * 2) + 12
    panel_h = max(left.height, right.height) + text_h
    panel = Image.new("RGB", (panel_w, panel_h), (8, 8, 8))
    panel.paste(left, (0, 0))
    panel.paste(right, (thumb_size[0] + 12, 0))
    draw = ImageDraw.Draw(panel)
    delta = float(morphology_row["dice"]) - float(threshold_row["dice"])
    lines = [
        "green=truth only red=prediction only yellow=overlap",
        f"{sid}: IoU threshold={float(threshold_row['iou']):.3f}, morph={float(morphology_row['iou']):.3f}; Dice delta={delta:+.3f}",
        f"truth_area={int(morphology_row['truth_area_px'])} pred_area threshold={int(threshold_row['pred_area_px'])} morph={int(morphology_row['pred_area_px'])}",
    ]
    y = max(left.height, right.height) + 6
    for line in lines:
        draw.text((4, y), line, fill=(230, 230, 230))
        y += 16
    return panel


def make_comparison_sheet(
    manifest_path: Path,
    dataset_root: Path,
    threshold_report_path: Path,
    morphology_report_path: Path,
    out_png: Path,
    out_json: Path,
    limit: int = 6,
    cols: int = 1,
) -> dict[str, Any]:
    manifest = {r["sequence_id"]: r for r in load_manifest(manifest_path)}
    threshold_report = load_json(threshold_report_path)
    morphology_report = load_json(morphology_report_path)
    threshold = row_map(threshold_report)
    morph = row_map(morphology_report)
    selected = select_sequences(threshold_report, morphology_report, limit=limit)
    panels: list[Image.Image] = []
    rows: list[dict[str, Any]] = []
    for sid in selected:
        record = manifest.get(sid)
        if record is None:
            continue
        panels.append(make_sequence_panel(record, threshold[sid], morph[sid], dataset_root))
        rows.append(
            {
                "sequence_id": sid,
                "threshold_iou": threshold[sid]["iou"],
                "threshold_dice": threshold[sid]["dice"],
                "morphology_iou": morph[sid]["iou"],
                "morphology_dice": morph[sid]["dice"],
                "dice_delta": float(morph[sid]["dice"]) - float(threshold[sid]["dice"]),
            }
        )
    if not panels:
        raise ValueError("no DIAS panels could be generated")
    cols = max(1, cols)
    n_rows = (len(panels) + cols - 1) // cols
    panel_w = max(p.width for p in panels)
    panel_h = max(p.height for p in panels)
    title_h = 52
    sheet = Image.new("RGB", (panel_w * cols, title_h + panel_h * n_rows), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    title = f"DIAS {morphology_report['eval_split']} vessel baseline comparison"
    subtitle = (
        f"threshold Dice={threshold_report['aggregate']['mean_dice']:.3f}, "
        f"morphology Dice={morphology_report['aggregate']['mean_dice']:.3f}; "
        "DIAS vessel masks only, no catheter/device labels"
    )
    draw.text((6, 6), title, fill=(255, 255, 255))
    draw.text((6, 26), subtitle, fill=(220, 220, 220))
    for idx, panel in enumerate(panels):
        x = (idx % cols) * panel_w
        y = title_h + (idx // cols) * panel_h
        sheet.paste(panel, (x, y))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    metadata = {
        "figure": str(out_png),
        "manifest": str(manifest_path),
        "dataset_root": str(dataset_root),
        "threshold_report": str(threshold_report_path),
        "morphology_report": str(morphology_report_path),
        "selected": rows,
        "color_key": {
            "green": "ground-truth vessel only",
            "red": "prediction only",
            "yellow": "truth/prediction overlap",
        },
    }
    out_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("research/synthetic_dsa/outputs/manifests/dias_manifest.jsonl"))
    ap.add_argument("--dataset-root", type=Path, default=Path("research/synthetic_dsa/data/dias/DIAS"))
    ap.add_argument("--threshold-report", type=Path, required=True)
    ap.add_argument("--morphology-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--cols", type=int, default=1)
    args = ap.parse_args()
    metadata = make_comparison_sheet(
        args.manifest,
        args.dataset_root,
        args.threshold_report,
        args.morphology_report,
        args.out,
        args.out_json,
        limit=args.limit,
        cols=args.cols,
    )
    print(json.dumps({"figure": metadata["figure"], "selected_count": len(metadata["selected"]), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
