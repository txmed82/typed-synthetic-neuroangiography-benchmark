# Synthetic Neuroangiography as a Testbed for Catheter Tracking and Procedural State Perception

## Working contribution

We introduce a typed synthetic DSA benchmark harness for procedural perception in neuroangiography. The benchmark represents vascular geometry, projection view, bolus dynamics, device overlays, and failure modes as explicit provenance-bearing artifacts, then evaluates models on vessel segmentation, catheter-tip localization, bolus phase estimation, and ambiguity/failure detection.

## Current verified artifacts

- Manifest schema: `research/synthetic_dsa/schemas/manifest.schema.json`
- Dependency-free manifest validator: `research/synthetic_dsa/scripts/validate_manifest.py`
- Contract fixture: `research/synthetic_dsa/outputs/manifests/smoke_manifest.jsonl`

## Figure plan

1. Typed pipeline diagram: `VascularGraph -> ProjectionView -> BolusCurve -> DSAFrameSequence -> DeviceOverlay -> VerifierResult`.
2. Synthetic sequence examples across AP/lateral/oblique projections.
3. Catheter tip labels, visibility states, and ambiguity cases.
4. Benchmark task grid.
5. Beyond-Dice failure examples: high mask overlap but poor tip/centerline/procedural state.
6. DIAS compatibility/sanity check.
7. Experiment manager loop and typed revision log.

## Methods TODO

- Implement CPU-only generator for a 10-sequence smoke pack.
- Add first verifier metrics: frame-count consistency, tip pixel error, bolus phase accuracy, Dice/IoU, and temporal consistency.
- Produce first figure pack from generated arrays, not placeholders.

## Results TODO

No model or generator results yet. Do not make claims beyond schema/contract validation until generated sequences and metrics exist.

## Limitations TODO

- Synthetic DSA realism must be validated against public DSA data such as DIAS where comparable.
- Public catheter-tip labels may not exist; catheter/device tasks may initially be synthetic-only.
- No clinical performance claims.
