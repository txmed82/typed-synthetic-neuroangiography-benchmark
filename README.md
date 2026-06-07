# Synthetic DSA Research Harness

Research-only workspace for Seldinger Labs track #3: **synthetic neuroangiography as a typed benchmark for catheter tracking and procedural state perception**.

This folder is intentionally isolated from the commercial TopCoW/aneurysm MVP. Do not un-park or wire this into production without explicit Colin approval.

## Current scope

Minimal arXiv-caliber benchmark artifact, built iteratively:

1. Typed synthetic DSA manifest schema.
2. CPU-only generator smoke tests for small DSA-like sequences.
3. Verifier metrics beyond Dice/IoU: catheter-tip error, bolus phase, temporal consistency, ambiguity/failure tags.
4. Simple baselines before any GPU jobs.
5. Paper skeleton and figure pack updated as experiments mature.

## Artifact contract

Every experiment should produce, when applicable:

- `outputs/manifests/*.jsonl` — one manifest record per generated sequence.
- `outputs/reports/*.json` — metrics/verifier outputs.
- `outputs/figures/*` — example panels or plots.
- `experiments.jsonl` — append-only experiment registry.
- `paper/*` — paper sections or figure TODOs updated from verified outputs only.

## Budget guardrail

Approved external compute budget: **$100 total**. Prefer local CPU work. Do not start RunPod/Modal/GPU work unless bounded, estimated, and checked against `/Users/colin/.hermes/data/seldinger_research_budget.json`.

## First benchmark tasks

- Vessel mask sequence segmentation.
- Catheter/wire tip localization.
- Bolus phase estimation.
- Ambiguity/failure detection under projection overlap, low contrast, motion/noise, and device-vessel relationship shifts.
