# Seldinger-DSA v0.1 release candidate

Research-only benchmark release candidate for the Seldinger Synthetic DSA / endovascular procedural-perception benchmark.

## Scope

This v0.1 package supports four controlled synthetic benchmark tasks:

- vessel segmentation over time
- catheter/wire tip localization
- bolus phase estimation
- failure/regime-shift analysis

It also uses DIAS as an external DSA vessel-segmentation sanity check. DIAS is vessel-mask only in this harness; it does not validate catheter-tip or device-state perception.

## Defensible claim

Use this release to support the paper claim that typed synthetic DSA artifacts and verifier metrics expose controlled catheter/projection failure modes that aggregate segmentation-only metrics can miss.

Do not claim clinical utility, procedural safety, diagnostic performance, or state-of-the-art DIAS performance from this release.

## Frozen synthetic schema

- Schema file: `schemas/manifest.schema.json`
- Schema version: `0.1.0`
- Sequence IDs: `sdsa_<tag>_<index>`
- Core typed fields:
  - `vascular_graph`
  - `projection_view`
  - `bolus_curve`
  - `dsa_frame_sequence`
  - `vessel_mask_sequence`
  - `device_mask_sequence` for v2+ manifests
  - `catheter_path`
  - `catheter_tip_state`
  - `device_vessel_relationship`
  - `appearance_model`
  - `failure_modes`
  - `benchmark_tasks`

## Frozen local manifests

Synthetic:

- `outputs/manifests/toy_v2_60_manifest.jsonl`
- `outputs/manifests/toy_v2_train42_manifest.jsonl`
- `outputs/manifests/toy_v2_eval18_manifest.jsonl`
- `outputs/manifests/toy_v3_60_manifest.jsonl`
- `outputs/manifests/toy_v3_train42_manifest.jsonl`
- `outputs/manifests/toy_v3_eval18_manifest.jsonl`
- `outputs/manifests/toy_v4_60_manifest.jsonl`
- `outputs/manifests/toy_v4_train42_manifest.jsonl`
- `outputs/manifests/toy_v4_eval18_manifest.jsonl`

External sanity check:

- `outputs/manifests/dias_manifest.jsonl`

## Deterministic regeneration commands

Run from repo root: `/Users/colin/Desktop/projects/seldinger`.

Generate v2/v3/v4 synthetic manifests:

```bash
python3 research/synthetic_dsa/scripts/generate_toy_sequences.py \
  --out research/synthetic_dsa/outputs --count 60 --seed 20260605 \
  --size 128 --frames 6 --realism v2 --tag toy_v2

python3 research/synthetic_dsa/scripts/generate_toy_sequences.py \
  --out research/synthetic_dsa/outputs --count 60 --seed 20260605 \
  --size 128 --frames 6 --realism v3 --tag toy_v3

python3 research/synthetic_dsa/scripts/generate_toy_sequences.py \
  --out research/synthetic_dsa/outputs --count 60 --seed 20260605 \
  --size 128 --frames 6 --realism v4 --tag toy_v4
```

Note: generated manifest records include `created_at` and `git_commit`, so byte-for-byte equality is not expected after regeneration. The deterministic content contract is the sequence geometry/pixels for a fixed seed, generator version, size, frames, realism, and tag.

## Validation and QA commands

Validate manifests:

```bash
for manifest in \
  research/synthetic_dsa/outputs/manifests/toy_v2_60_manifest.jsonl \
  research/synthetic_dsa/outputs/manifests/toy_v3_60_manifest.jsonl \
  research/synthetic_dsa/outputs/manifests/toy_v4_60_manifest.jsonl; do
  python3 research/synthetic_dsa/scripts/validate_manifest.py "$manifest"
done
```

Run geometry QA for typed device/catheter artifacts:

```bash
python3 research/synthetic_dsa/scripts/audit_manifest_geometry.py \
  research/synthetic_dsa/outputs/manifests/toy_v2_60_manifest.jsonl \
  --root research/synthetic_dsa \
  --out-json research/synthetic_dsa/outputs/reports/toy_v2_60_geometry_qa_report.json

python3 research/synthetic_dsa/scripts/audit_manifest_geometry.py \
  research/synthetic_dsa/outputs/manifests/toy_v3_60_manifest.jsonl \
  --root research/synthetic_dsa \
  --out-json research/synthetic_dsa/outputs/reports/toy_v3_60_geometry_qa_report.json

python3 research/synthetic_dsa/scripts/audit_manifest_geometry.py \
  research/synthetic_dsa/outputs/manifests/toy_v4_60_manifest.jsonl \
  --root research/synthetic_dsa \
  --out-json research/synthetic_dsa/outputs/reports/toy_v4_60_geometry_qa_report.json
```

Run current synthetic baseline comparison:

```bash
python3 research/synthetic_dsa/scripts/run_tiny_cpu_baseline.py \
  research/synthetic_dsa/outputs/manifests/toy_v4_eval18_manifest.jsonl \
  --root research/synthetic_dsa \
  --train-manifest research/synthetic_dsa/outputs/manifests/toy_v2_train42_manifest.jsonl \
  --train-manifest research/synthetic_dsa/outputs/manifests/toy_v3_train42_manifest.jsonl \
  --baseline patch_ranker_dp --phase-rule temporal_rank \
  --out-json research/synthetic_dsa/outputs/reports/cross_mixed_v2v3train_v4eval_tiny_patch_ranker_dp_temporal_cpu_report.json \
  --out-md research/synthetic_dsa/outputs/reports/cross_mixed_v2v3train_v4eval_tiny_patch_ranker_dp_temporal_cpu_report.md

python3 research/synthetic_dsa/scripts/run_tiny_cpu_baseline.py \
  research/synthetic_dsa/outputs/manifests/toy_v4_eval18_manifest.jsonl \
  --root research/synthetic_dsa \
  --train-manifest research/synthetic_dsa/outputs/manifests/toy_v4_train42_manifest.jsonl \
  --baseline patch_ranker_dp --phase-rule temporal_rank \
  --out-json research/synthetic_dsa/outputs/reports/within_v4train_v4eval_tiny_patch_ranker_dp_temporal_cpu_report.json \
  --out-md research/synthetic_dsa/outputs/reports/within_v4train_v4eval_tiny_patch_ranker_dp_temporal_cpu_report.md
```

Run DIAS vessel-only sanity baselines:

```bash
python3 research/synthetic_dsa/scripts/run_dias_segmentation_baseline.py \
  research/synthetic_dsa/outputs/manifests/dias_manifest.jsonl \
  --dataset-root research/synthetic_dsa/data/dias/DIAS \
  --train-split training --eval-split validation \
  --baseline projection_morphology \
  --out-json research/synthetic_dsa/outputs/reports/dias_validation_projection_morphology_report.json \
  --out-md research/synthetic_dsa/outputs/reports/dias_validation_projection_morphology_report.md

python3 research/synthetic_dsa/scripts/run_dias_segmentation_baseline.py \
  research/synthetic_dsa/outputs/manifests/dias_manifest.jsonl \
  --dataset-root research/synthetic_dsa/data/dias/DIAS \
  --train-split training --eval-split test \
  --baseline projection_morphology \
  --out-json research/synthetic_dsa/outputs/reports/dias_test_projection_morphology_report.json \
  --out-md research/synthetic_dsa/outputs/reports/dias_test_projection_morphology_report.md
```

## Current key result

The current v0.1 controlled-shift result is v4 coil/projection stress:

- mixed v2+v3 train -> v4 eval: mean IoU 0.481, mean Dice 0.622, mean tip error 2.154 px, within-2px tip rate 0.556
- v4 train -> v4 eval: mean IoU 0.649, mean Dice 0.755, mean tip error 1.539 px, within-2px tip rate 0.824
- v4 geometry QA pass rate: 1.0

Interpretation: in-domain v4 training recovers segmentation and strict tip precision relative to mixed v2/v3 training under the v4 coil/projection stress protocol. This is the best current publishable research story.

## Current DIAS sanity result

DIAS projection-morphology baseline:

- validation: mean IoU 0.420, mean Dice 0.578
- test: mean IoU 0.459, mean Dice 0.612

Interpretation: DIAS wiring and vessel-mask evaluation are functional. This is a weak classical sanity baseline, not a SOTA comparison and not device validation.

## Paper artifacts

- Draft: `paper/main.tex`
- Readiness tracker: `paper/preprint_readiness.md`
- Metrics summary: `outputs/reports/preprint_metrics_summary.json`
- DIAS paper figure: `outputs/figures/dias_test_threshold_vs_morphology_paper.png`

## Release caveats

- This is a local release candidate, not a public data release yet.
- Licensing is intentionally conservative until Colin approves public release scope.
- Synthetic generator realism remains toy/procedural; the publishable value is typed perturbation + verifier design, not photorealism.
- DIAS labels are vessel-only.
- No cloud/GPU spend was required for this release candidate.
