#!/usr/bin/env python3
"""Upload the prepared Seldinger-DSA v0.1 generated-artifact archive to Zenodo.

Usage:
  ZENODO_ACCESS_TOKEN=... python scripts/upload_zenodo_artifacts.py --publish

The script never stores or prints the token. It writes the resulting record metadata to
`zenodo_record_v0.1.json` on success.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import requests

BASE_DIR = Path(__file__).resolve().parents[1]
ZENODO_STAGING = BASE_DIR / "zenodo_staging"
ARCHIVE = ZENODO_STAGING / "seldinger_dsa_v0.1_generated_artifacts.zip"
ARCHIVE_MANIFEST = ZENODO_STAGING / "seldinger_dsa_v0.1_generated_artifacts_manifest.json"
OUT_RECORD = BASE_DIR / "zenodo_record_v0.1.json"


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def request_or_fail(method: str, url: str, *, headers: dict, **kwargs) -> requests.Response:
    response = requests.request(method, url, headers=headers, timeout=kwargs.pop("timeout", 60), **kwargs)
    if response.status_code >= 300:
        fail(f"{method} {url} failed: HTTP {response.status_code} {response.text[:1000]}")
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org instead of zenodo.org")
    parser.add_argument("--publish", action="store_true", help="Publish the deposition after upload")
    args = parser.parse_args()

    token = os.environ.get("ZENODO_ACCESS_TOKEN")
    if not token:
        fail("Set ZENODO_ACCESS_TOKEN in the environment.")
    if not ARCHIVE.exists() or not ARCHIVE_MANIFEST.exists():
        fail(f"Missing prepared archive or manifest under {ZENODO_STAGING}")

    api_base = "https://sandbox.zenodo.org/api" if args.sandbox else "https://zenodo.org/api"
    headers = {"Authorization": f"Bearer {token}"}
    metadata = {
        "metadata": {
            "title": "Typed Synthetic Neuroangiography Benchmark v0.1 generated artifacts",
            "upload_type": "dataset",
            "description": (
                "Generated artifacts for the typed synthetic neuroangiography benchmark v0.1 research release candidate: "
                "synthetic DSA sequences, manifests, schema, experiment reports, generated figures, "
                "and derived DIAS prediction overlays. The original DIAS dataset is not redistributed; "
                "cite DIAS separately at https://doi.org/10.5281/zenodo.11396520. This record is "
                "restricted pending final public-release/license approval."
            ),
            "creators": [{"name": "Son, Colin", "affiliation": "Seldinger, Inc."}],
            "access_right": "restricted",
            "access_conditions": (
                "Research release candidate archived for DOI/provenance. Access and reuse are subject "
                "to author approval and final license terms. Third-party DIAS data are not included and "
                "remain governed by their original license/citation requirements."
            ),
            "keywords": [
                "digital subtraction angiography",
                "DSA",
                "endovascular",
                "synthetic data",
                "benchmark",
                "catheter tip localization",
                "vessel segmentation",
                "DIAS",
            ],
            "prereserve_doi": True,
            "related_identifiers": [
                {"relation": "isSupplementTo", "identifier": "https://github.com/txmed82/typed-synthetic-neuroangiography-benchmark", "scheme": "url"},
                {"relation": "cites", "identifier": "10.5281/zenodo.11396520", "scheme": "doi"},
                {"relation": "cites", "identifier": "10.1016/j.media.2024.103247", "scheme": "doi"},
            ],
        }
    }

    deposition = request_or_fail("POST", f"{api_base}/deposit/depositions", headers=headers, json=metadata).json()
    bucket = deposition["links"]["bucket"]
    for file_path in [ARCHIVE, ARCHIVE_MANIFEST]:
        with file_path.open("rb") as fh:
            request_or_fail("PUT", f"{bucket}/{file_path.name}", headers=headers, data=fh, timeout=300)

    record = deposition
    if args.publish:
        record = request_or_fail("POST", deposition["links"]["publish"], headers=headers).json()

    output = {
        "id": record.get("id"),
        "conceptrecid": record.get("conceptrecid"),
        "doi": record.get("doi"),
        "conceptdoi": record.get("conceptdoi"),
        "record_url": record.get("links", {}).get("html"),
        "submitted": record.get("submitted"),
        "state": record.get("state"),
        "files": [
            {"key": file.get("key"), "size": file.get("size"), "checksum": file.get("checksum")}
            for file in record.get("files", [])
        ],
    }
    OUT_RECORD.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
