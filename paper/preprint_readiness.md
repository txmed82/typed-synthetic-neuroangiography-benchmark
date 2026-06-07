# Path to an arXiv-ready Seldinger Endovascular Benchmark preprint

## Working title
A Typed Synthetic Neuroangiography Benchmark for Endovascular Procedural Perception

## Current status
The project has crossed from concept into a reproducible benchmark prototype. It has a typed manifest, generator regimes v0-v4, geometry QA, verifier metrics, local baselines, failure overlays, and a DIAS external vessel-segmentation sanity check.

## Minimum bar for a credible first arXiv preprint

### Must have before posting
1. **Frozen benchmark release v0.1**
   - Freeze schema, generation seeds, v4 stress protocol, train/eval splits, and validation commands.
   - Add `CITATION.cff`, `LICENSE`, and a reproducibility README.
2. **Stronger DIAS real-data sanity baseline** — **initial local version complete**
   - Added a projection-morphology baseline on top of learned DIAS temporal projection thresholding.
   - Validation/test reports and overlay figures exist; still not a SOTA comparison.
3. **Synthetic-to-real relevance experiment** — **initial local version complete with modest positive result**
   - Manuscript now includes a narrow construct-validity claim: DIAS validates only the real-vessel portion of the pipeline, while Seldinger-DSA supplies synthetic procedural/device labels absent from public real data.
   - Implemented `scripts/run_synthetic_to_dias_vessel_transfer.py`.
   - Synthetic v2/v3/v4 train masks estimate vessel occupancy priors; each prior is applied as an adaptive top-percentile rule on DIAS temporal-range projections with fixed small-component removal.
   - Best synthetic variant by validation Dice: v2 area prior.
   - Result: DIAS test Dice improves from 0.612 projection-threshold / 0.612 projection-morphology to 0.624 synthetic v2 area-prior; small but positive.
   - Do not claim real catheter/device performance without labeled real or phantom catheterized DSA.
4. **Figure pack**
   - Pipeline diagram.
   - v4 stress contact sheet.
   - Failure overlays with truth/prediction/error vector.
   - DIAS prediction overlay examples — **draft validation/test sheets generated**.
   - Results panel: regime shift and DIAS sanity numbers.
5. **Citation verification** — **core draft citations verified**
   - Verified and replaced placeholder BibTeX for DIAS, MedSAM2, AI Scientist-v2, and Self-Revising Discovery Systems using arXiv/source pages.
   - Do not cite unverifiable Sophon snippets unless tied to arXiv/DOI/source pages.
6. **Limitations section strong enough for medical AI review**
   - Synthetic-only catheter-tip labels.
   - Toy/procedural generator realism limitations.
   - DIAS vessel-only sanity check, not device validation.
   - No clinical utility/safety claims.

### Nice-to-have before posting
- A tiny U-Net/SegFormer or SAM2/MedSAM2 comparison on synthetic vessel segmentation.
- A small VLM/human-style visual realism review protocol.
- A public repo branch with data-download instructions and deterministic regeneration commands.

## Recommended execution order
1. Freeze v0.1 schema/splits and write reproducibility README.
2. Add DIAS morphology/connected-component baseline and DIAS overlay figures.
3. Synthetic-to-DIAS vessel adaptation result is now complete; add a final figure panel for it.
4. Generate final figure pack.
5. Complete LaTeX draft and bibliography.
6. Internal red-team review: novelty, synthetic realism, overclaiming, reproducibility.
7. Post as arXiv preprint only if limitations and code release are clean.

## Preprint claim to defend
This paper should **not** claim clinical performance. The defensible claim is:

> We provide a reproducible, typed synthetic DSA benchmark harness for controlled evaluation of procedural perception tasks that are difficult to study with public data, and we show that typed regime shifts expose catheter-tip/projection failures not captured by segmentation-only metrics. DIAS is used as an external real-vessel sanity check; a synthetic v2 vessel-occupancy prior gives a modest positive DIAS vessel-mask adaptation result, but DIAS does not validate real catheter/device perception.
