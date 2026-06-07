# Zenodo archive contents for typed synthetic neuroangiography benchmark v0.1

Canonical generated-data archive:
- Record: https://zenodo.org/records/20581909
- DOI: https://doi.org/10.5281/zenodo.20581909
- Concept DOI: https://doi.org/10.5281/zenodo.20581908

This archive is for generated benchmark/data artifacts, not source code.

Included in Zenodo:
- `outputs/sequences/` — generated synthetic DSA sequence artifacts used by the v0.1 benchmark.
- `outputs/manifests/` — JSONL manifests for synthetic datasets, splits, DIAS adapter manifests, and smoke/test manifests.
- `outputs/reports/` — machine-readable and human-readable experiment reports used by the manuscript.
- `outputs/figures/` — generated figure panels and failure-case overlays used for inspection/manuscript packaging.
- `outputs/dias_predictions/` — derived DIAS prediction masks/overlays from the local sanity-check experiments.
- `schemas/` — manifest schema used to validate benchmark records.
- `release_manifest_v0.1.json` — hash/record-count manifest for the release candidate.
- `RELEASE_v0.1.md`, `README.md`, `CITATION.cff`, `NOTICE`, `LICENSE` — release-candidate metadata and conservative legal notices.

Excluded from Zenodo:
- `data/dias/DIAS.zip` and extracted DIAS source data. DIAS is third-party data and remains governed by its original DOI/license: https://doi.org/10.5281/zenodo.11396520.
- Source code/scripts/tests/manuscript source. Those belong in GitHub.
- Local caches, Python bytecode, and build artifacts.

Access note:
This v0.1 archive is a restricted research release candidate. It is archived for DOI/provenance, but public reuse/licensing should remain gated until final author/license approval.
