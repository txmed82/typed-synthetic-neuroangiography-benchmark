#!/usr/bin/env python3
"""Generator-seed stability experiment for Seldinger-DSA synthetic regimes."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
MANIFESTS = ROOT / "outputs" / "manifests"
GEN = ROOT / "scripts" / "generate_toy_sequences.py"
BASE = ROOT / "scripts" / "run_tiny_cpu_baseline.py"

SEEDS = [20260621, 20260622, 20260623]
REALISMS = ["v2", "v3", "v4"]


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def split_manifest(tag: str) -> tuple[Path, Path]:
    full = MANIFESTS / f"{tag}_60_manifest.jsonl"
    rows = load_jsonl(full)
    train = MANIFESTS / f"{tag}_train42_manifest.jsonl"
    evalp = MANIFESTS / f"{tag}_eval18_manifest.jsonl"
    write_jsonl(train, rows[:42])
    write_jsonl(evalp, rows[42:])
    return train, evalp


def metric_row(label: str, path: Path, seed: int) -> dict:
    report = json.loads(path.read_text())
    agg = report["aggregate"]
    return {
        "seed": seed,
        "label": label,
        "report": str(path),
        "mean_iou": agg["mean_iou"],
        "mean_dice": agg["mean_dice"],
        "mean_tip_error_px": agg["mean_tip_error_px"],
        "tip_within_2px_rate": agg["tip_within_2px_rate"],
        "tip_within_5px_rate": agg["tip_within_5px_rate"],
        "phase_accuracy": agg["phase_accuracy"],
    }


def ci_min_max(vals: list[float]) -> dict:
    return {"mean": mean(vals), "min": min(vals), "max": max(vals), "n": len(vals)}


def main() -> int:
    rows = []
    for seed in SEEDS:
        tags = {}
        for realism in REALISMS:
            tag = f"toy_{realism}_seed{seed}"
            tags[realism] = tag
            manifest = MANIFESTS / f"{tag}_60_manifest.jsonl"
            if not manifest.exists():
                run(["python3", str(GEN), "--out", str(ROOT / "outputs"), "--count", "60", "--seed", str(seed + {"v2": 0, "v3": 1000, "v4": 2000}[realism]), "--size", "128", "--frames", "6", "--realism", realism, "--tag", tag])
            split_manifest(tag)
        v2_train = MANIFESTS / f"{tags['v2']}_train42_manifest.jsonl"
        v3_train = MANIFESTS / f"{tags['v3']}_train42_manifest.jsonl"
        v4_train = MANIFESTS / f"{tags['v4']}_train42_manifest.jsonl"
        v4_eval = MANIFESTS / f"{tags['v4']}_eval18_manifest.jsonl"
        runs = [
            ("mixed_v2v3_to_v4", ["--train-manifest", str(v2_train), "--train-manifest", str(v3_train)], v4_eval),
            ("v4_to_v4", ["--train-manifest", str(v4_train)], v4_eval),
        ]
        for label, train_args, eval_manifest in runs:
            out_json = REPORTS / f"seed_stability_{label}_{seed}_patch_ranker_dp.json"
            out_md = REPORTS / f"seed_stability_{label}_{seed}_patch_ranker_dp.md"
            if not out_json.exists():
                run(["python3", str(BASE), str(eval_manifest), "--root", str(ROOT), *train_args, "--phase-rule", "temporal_rank", "--baseline", "patch_ranker_dp", "--out-json", str(out_json), "--out-md", str(out_md)])
            rows.append(metric_row(label, out_json, seed))
    by_label = {}
    for label in sorted(set(r["label"] for r in rows)):
        lr = [r for r in rows if r["label"] == label]
        by_label[label] = {k: ci_min_max([float(r[k]) for r in lr]) for k in ["mean_dice", "mean_iou", "mean_tip_error_px", "tip_within_2px_rate", "tip_within_5px_rate", "phase_accuracy"]}
    # Paired per-seed deltas: in-domain v4 minus cross mixed→v4.
    deltas = []
    for seed in SEEDS:
        cross = next(r for r in rows if r["seed"] == seed and r["label"] == "mixed_v2v3_to_v4")
        within = next(r for r in rows if r["seed"] == seed and r["label"] == "v4_to_v4")
        deltas.append({
            "seed": seed,
            "delta_v4_within_minus_cross_tip_at_2": within["tip_within_2px_rate"] - cross["tip_within_2px_rate"],
            "delta_v4_within_minus_cross_dice": within["mean_dice"] - cross["mean_dice"],
            "delta_v4_within_minus_cross_tip_error_px": within["mean_tip_error_px"] - cross["mean_tip_error_px"],
        })
    delta_summary = {k: ci_min_max([d[k] for d in deltas]) for k in ["delta_v4_within_minus_cross_tip_at_2", "delta_v4_within_minus_cross_dice", "delta_v4_within_minus_cross_tip_error_px"]}
    report = {"seeds": SEEDS, "rows": rows, "by_label": by_label, "paired_deltas": deltas, "paired_delta_summary": delta_summary}
    out_json = REPORTS / "generator_seed_stability_patch_ranker_dp_report.json"
    out_md = REPORTS / "generator_seed_stability_patch_ranker_dp_report.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Generator-seed stability: patch-ranker DP", "", "Three independent generator seeds; 60 sequences/regime/seed; first 42 train, last 18 eval.", ""]
    for label, stats in by_label.items():
        lines.append(f"## {label}")
        for k, v in stats.items():
            lines.append(f"- {k}: mean={v['mean']:.4f}, range=[{v['min']:.4f}, {v['max']:.4f}], n={v['n']}")
        lines.append("")
    lines.append("## Paired in-domain v4 minus mixed v2+v3→v4 deltas")
    for k, v in delta_summary.items():
        lines.append(f"- {k}: mean={v['mean']:+.4f}, range=[{v['min']:+.4f}, {v['max']:+.4f}], n={v['n']}")
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "paired_delta_summary": delta_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
