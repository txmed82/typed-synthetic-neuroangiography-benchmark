#!/usr/bin/env python3
"""Create a contact-sheet figure for toy synthetic DSA montages."""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageDraw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", type=Path, default=Path("research/synthetic_dsa/outputs/figures"))
    ap.add_argument("--out", type=Path, default=Path("research/synthetic_dsa/outputs/figures/toy_contact_sheet.png"))
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--pattern", default="sdsa_toy_*_montage.png")
    args = ap.parse_args()
    paths = sorted(args.figures.glob(args.pattern))[: args.max]
    if not paths:
        raise SystemExit("no toy montage figures found")
    thumbs = []
    for p in paths:
        img = Image.open(p).convert("L")
        img.thumbnail((384, 64))
        canvas = Image.new("L", (420, 88), 0)
        canvas.paste(img, (8, 8))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 72), p.stem.replace("_montage", ""), fill=220)
        thumbs.append(canvas)
    cols = 1
    sheet = Image.new("L", (420 * cols, 88 * len(thumbs)), 0)
    for i, t in enumerate(thumbs):
        sheet.paste(t, (0, i * 88))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
