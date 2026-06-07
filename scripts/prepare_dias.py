#!/usr/bin/env python3
"""Prepare DIAS public DSA dataset manifests for Seldinger external sanity checks.

DIAS layout expected after extracting DIAS.zip:
  training/images/image_s0_i0.png
  training/labels/label_s0.png
  validation/images/image_s30_i0.png
  validation/labels/label_s30.png
  test/images/image_s40_i0.png
  test/labels/label_s40.png
  unlabeled_DSA/images/image_s100_i0.png

This adapter is intentionally dependency-light. It does not alter DIAS pixels; it
writes a JSONL manifest that points to the extracted files.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DIAS_ZENODO_URL = "https://zenodo.org/records/11401368/files/DIAS.zip?download=1"
IMAGE_RE = re.compile(r"^image_s(?P<seq>[^_]+)_i(?P<frame>\d+)\.png$")
LABEL_RE = re.compile(r"^label_s(?P<seq>[^.]+)\.png$")
DEFAULT_SPLITS = ["training", "validation", "test", "unlabeled_DSA"]


def parse_dias_image_name(name: str) -> tuple[str, int]:
    match = IMAGE_RE.match(Path(name).name)
    if not match:
        raise ValueError(f"not a DIAS image filename: {name}")
    return match.group("seq"), int(match.group("frame"))


def parse_dias_label_name(name: str) -> str:
    match = LABEL_RE.match(Path(name).name)
    if not match:
        raise ValueError(f"not a DIAS label filename: {name}")
    return match.group("seq")


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.size


def split_dirs(dataset_root: Path, split: str) -> tuple[Path, Path | None]:
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"
    return image_dir, label_dir if label_dir.exists() else None


def build_manifest_records(dataset_root: Path, splits: list[str] | None = None) -> list[dict[str, Any]]:
    dataset_root = dataset_root.resolve()
    splits = splits or DEFAULT_SPLITS
    records: list[dict[str, Any]] = []
    created_at = datetime.now(timezone.utc).isoformat()

    for split in splits:
        image_dir, label_dir = split_dirs(dataset_root, split)
        if not image_dir.exists():
            continue
        grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
        for image_path in sorted(image_dir.glob("image_s*_i*.png")):
            seq, frame_idx = parse_dias_image_name(image_path.name)
            grouped[seq].append((frame_idx, image_path))
        label_by_seq: dict[str, Path] = {}
        if label_dir and label_dir.exists():
            for label_path in sorted(label_dir.glob("label_s*.png")):
                label_by_seq[parse_dias_label_name(label_path.name)] = label_path

        for seq, frames in sorted(grouped.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
            frames = sorted(frames)
            frame_paths = [p for _, p in frames]
            width, height = image_size(frame_paths[0])
            label_path = label_by_seq.get(seq)
            has_label = label_path is not None
            record: dict[str, Any] = {
                "schema_version": "dias-adapter-0.1.0",
                "source_dataset": "DIAS",
                "source_url": "https://zenodo.org/records/11401368",
                "sequence_id": f"dias_s{seq}",
                "dias_sequence": seq,
                "split": split,
                "created_at": created_at,
                "dsa_frame_sequence": {
                    "uri": rel(image_dir, dataset_root),
                    "frame_count": len(frame_paths),
                    "height": height,
                    "width": width,
                    "dtype": "uint8",
                },
                "frame_files": [rel(p, dataset_root) for p in frame_paths],
                "vessel_mask_sequence": {
                    "uri": rel(label_path, dataset_root) if label_path else None,
                    "frame_count": 1 if label_path else 0,
                    "height": height,
                    "width": width,
                    "dtype": "bool",
                    "label_semantics": "sequence_level_intracranial_artery_mask",
                },
                "benchmark_tasks": ["vessel_segmentation"] if has_label else [],
                "has_labels": has_label,
            }
            records.append(record)
    return records


def write_manifest(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def download_and_extract(out_dir: Path, url: str = DIAS_ZENODO_URL, force: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "DIAS.zip"
    dataset_root = out_dir / "DIAS"
    if dataset_root.exists() and not force:
        return dataset_root
    if force or not zip_path.exists():
        urllib.request.urlretrieve(url, zip_path)  # noqa: S310 - public fixed research dataset URL
    if dataset_root.exists() and force:
        import shutil

        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dataset_root)
    # Some archives may contain a nested top-level directory; normalize if needed.
    for candidate in [dataset_root, *[p for p in dataset_root.iterdir() if p.is_dir()]]:
        if (candidate / "training" / "images").exists() or (candidate / "test" / "images").exists():
            return candidate
    return dataset_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, help="Extracted DIAS root containing training/validation/test folders")
    ap.add_argument("--download-dir", type=Path, default=Path("research/synthetic_dsa/data/dias"))
    ap.add_argument("--download", action="store_true", help="Download DIAS.zip from Zenodo and extract it")
    ap.add_argument("--force", action="store_true", help="Force re-download/re-extract when --download is used")
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--out", type=Path, default=Path("research/synthetic_dsa/outputs/manifests/dias_manifest.jsonl"))
    args = ap.parse_args()

    dataset_root = args.dataset_root
    if args.download:
        dataset_root = download_and_extract(args.download_dir, force=args.force)
    if dataset_root is None:
        raise SystemExit("provide --dataset-root or --download")
    records = build_manifest_records(dataset_root, splits=args.splits)
    if not records:
        raise SystemExit(f"no DIAS records found under {dataset_root}")
    write_manifest(records, args.out)
    print(json.dumps({
        "dataset_root": str(Path(dataset_root).resolve()),
        "manifest": str(args.out),
        "records": len(records),
        "labeled_records": sum(1 for r in records if r.get("has_labels")),
        "splits": sorted({r["split"] for r in records}),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
