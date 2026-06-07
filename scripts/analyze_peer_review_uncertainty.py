#!/usr/bin/env python3
"""Peer-review uncertainty analysis for Seldinger-DSA v0.1.

Adds sequence-level uncertainty to the synthetic-to-DIAS vessel-prior result and
summarizes tip-baseline divergence from existing synthetic reports.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def ci(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "median": median(values), "ci95_low": pct(values, 0.025), "ci95_high": pct(values, 0.975)}


def per_sequence_dice(result: dict[str, Any], model: str, split: str) -> dict[str, float]:
    if model.startswith("synthetic_area_prior_"):
        rows = result["results"][f"{model}:{split}"]["per_sequence"]
    else:
        key = {
            ("DIAS projection-threshold", "validation"): "dias_projection_threshold_validation",
            ("DIAS projection-threshold", "test"): "dias_projection_threshold_test",
            ("DIAS projection-morphology", "validation"): "dias_projection_morphology_validation",
            ("DIAS projection-morphology", "test"): "dias_projection_morphology_test",
        }[(model, split)]
        rows = load_json(Path(result["baseline_reports"][key]))["per_sequence"]
    return {r["sequence_id"]: float(r["dice"]) for r in rows}


def bootstrap_mean(values: list[float], rng: random.Random, n_boot: int = 20000) -> list[float]:
    n = len(values)
    return [mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot)]


def dias_uncertainty() -> dict[str, Any]:
    rng = random.Random(20260606)
    report = load_json(REPORTS / "synthetic_to_dias_vessel_transfer_report.json")
    models = [row["model"] for row in report["comparison"]]
    split_model = {split: {model: per_sequence_dice(report, model, split) for model in models} for split in ["validation", "test"]}

    model_stats: dict[str, Any] = {}
    for split in ["validation", "test"]:
        for model in models:
            vals = list(split_model[split][model].values())
            model_stats[f"{model}:{split}"] = {"n": len(vals), **ci(bootstrap_mean(vals, rng))}

    pairwise: dict[str, Any] = {}
    for model in [m for m in models if m.startswith("synthetic_area_prior_")]:
        for baseline in ["DIAS projection-threshold", "DIAS projection-morphology"]:
            common = sorted(set(split_model["test"][model]) & set(split_model["test"][baseline]))
            deltas = [split_model["test"][model][sid] - split_model["test"][baseline][sid] for sid in common]
            boot = bootstrap_mean(deltas, rng)
            pairwise[f"{model} minus {baseline}:test"] = {
                "n": len(deltas),
                "point_mean_delta": mean(deltas),
                **ci(boot),
                "prob_delta_gt_0": sum(v > 0 for v in boot) / len(boot),
            }

    # Bootstrap the validation selection rule itself. For each validation resample,
    # choose the synthetic model with best resampled validation Dice, then report the
    # paired test delta vs morphology/threshold on a bootstrapped test sample.
    synthetic = [m for m in models if m.startswith("synthetic_area_prior_")]
    val_ids = sorted(next(iter(split_model["validation"].values())).keys())
    test_ids = sorted(next(iter(split_model["test"].values())).keys())
    selected = []
    selected_delta_threshold = []
    selected_delta_morphology = []
    for _ in range(20000):
        val_sample = [val_ids[rng.randrange(len(val_ids))] for _ in val_ids]
        scores = {m: mean(split_model["validation"][m][sid] for sid in val_sample) for m in synthetic}
        # deterministic tie-break by model name so fragility is visible in counts
        choice = max(sorted(scores), key=lambda m: scores[m])
        selected.append(choice)
        test_sample = [test_ids[rng.randrange(len(test_ids))] for _ in test_ids]
        selected_delta_threshold.append(mean(split_model["test"][choice][sid] - split_model["test"]["DIAS projection-threshold"][sid] for sid in test_sample))
        selected_delta_morphology.append(mean(split_model["test"][choice][sid] - split_model["test"]["DIAS projection-morphology"][sid] for sid in test_sample))

    selection_counts = Counter(selected)
    selection_fragility = {
        "validation_n": len(val_ids),
        "test_n": len(test_ids),
        "selection_probabilities": {k: v / len(selected) for k, v in sorted(selection_counts.items())},
        "selected_delta_vs_threshold_test": {**ci(selected_delta_threshold), "prob_delta_gt_0": sum(v > 0 for v in selected_delta_threshold) / len(selected_delta_threshold)},
        "selected_delta_vs_morphology_test": {**ci(selected_delta_morphology), "prob_delta_gt_0": sum(v > 0 for v in selected_delta_morphology) / len(selected_delta_morphology)},
    }
    return {"model_stats": model_stats, "pairwise_test_deltas": pairwise, "selection_fragility": selection_fragility}


def tip_baseline_divergence() -> dict[str, Any]:
    summary = load_json(REPORTS / "preprint_metrics_summary.json")
    rows = summary["rows"]
    selected = [r for r in rows if "→" in r["label"] and ("Tiny temporal" in r["label"] or "Patch-ranker DP" in r["label"])]
    by_regime: dict[str, dict[str, Any]] = {}
    for row in selected:
        label = row["label"]
        if ": " not in label:
            continue
        baseline, regime = label.split(": ", 1)
        by_regime.setdefault(regime, {})[baseline] = row
    divergence = []
    for regime, vals in sorted(by_regime.items()):
        if "Tiny temporal" in vals and "Patch-ranker DP" in vals:
            tiny = vals["Tiny temporal"]
            dp = vals["Patch-ranker DP"]
            divergence.append({
                "regime": regime,
                "tiny_tip_at_2": tiny["tip_within_2px_rate"],
                "patch_ranker_dp_tip_at_2": dp["tip_within_2px_rate"],
                "delta_tip_at_2": dp["tip_within_2px_rate"] - tiny["tip_within_2px_rate"],
                "tiny_mean_tip_error_px": tiny["mean_tip_error_px"],
                "patch_ranker_dp_mean_tip_error_px": dp["mean_tip_error_px"],
                "tiny_mean_dice": tiny["mean_dice"],
                "patch_ranker_dp_mean_dice": dp["mean_dice"],
            })
    return {"regime_comparisons": divergence}


def write_md(report: dict[str, Any], path: Path) -> None:
    lines = ["# Peer-review uncertainty analysis", "", "## DIAS vessel-prior uncertainty", ""]
    sf = report["dias_uncertainty"]["selection_fragility"]
    lines.append(f"Validation n={sf['validation_n']}; test n={sf['test_n']}.")
    lines.append("")
    lines.append("Selection probabilities under validation bootstrap:")
    for model, prob in sf["selection_probabilities"].items():
        lines.append(f"- {model}: {prob:.3f}")
    lines.append("")
    for label, stats in sf.items():
        if label.startswith("selected_delta"):
            lines.append(f"- {label}: mean={stats['mean']:+.4f}, 95% CI [{stats['ci95_low']:+.4f}, {stats['ci95_high']:+.4f}], P(delta>0)={stats['prob_delta_gt_0']:.3f}")
    lines.extend(["", "Pairwise test deltas:"])
    for label, stats in report["dias_uncertainty"]["pairwise_test_deltas"].items():
        lines.append(f"- {label}: n={stats['n']}, point={stats['point_mean_delta']:+.4f}, bootstrap mean={stats['mean']:+.4f}, 95% CI [{stats['ci95_low']:+.4f}, {stats['ci95_high']:+.4f}], P(delta>0)={stats['prob_delta_gt_0']:.3f}")
    lines.extend(["", "## Tip-baseline divergence", ""])
    for row in report["tip_baseline_divergence"]["regime_comparisons"]:
        lines.append(f"- {row['regime']}: Tiny Tip@2={row['tiny_tip_at_2']:.3f}; Patch-ranker DP Tip@2={row['patch_ranker_dp_tip_at_2']:.3f}; delta={row['delta_tip_at_2']:+.3f}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    report = {"dias_uncertainty": dias_uncertainty(), "tip_baseline_divergence": tip_baseline_divergence()}
    out_json = REPORTS / "peer_review_uncertainty_report.json"
    out_md = REPORTS / "peer_review_uncertainty_report.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_md(report, out_md)
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "selection": report["dias_uncertainty"]["selection_fragility"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
