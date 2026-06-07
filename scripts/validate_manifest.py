#!/usr/bin/env python3
"""Validate Seldinger synthetic DSA manifest JSONL files.

Dependency-free semantic checks for the v0.1 manifest contract. This is not a
full JSON Schema validator; it catches the invariants most likely to break
benchmark comparability before generated data exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

VALID_PHASES = {"precontrast", "arrival", "arterial_peak", "washout"}
VALID_TASKS = {
    "vessel_segmentation",
    "catheter_tip_localization",
    "bolus_phase_estimation",
    "failure_detection",
}

REQUIRED_TOP_LEVEL = [
    "schema_version",
    "sequence_id",
    "generator",
    "provenance",
    "vascular_graph",
    "projection_view",
    "bolus_curve",
    "dsa_frame_sequence",
    "vessel_mask_sequence",
    "catheter_path",
    "catheter_tip_state",
    "device_vessel_relationship",
    "failure_modes",
    "benchmark_tasks",
]


def fail(errors: list[str], sequence_id: str, message: str) -> None:
    errors.append(f"{sequence_id}: {message}")


def require_keys(errors: list[str], sequence_id: str, obj: dict[str, Any], keys: list[str], prefix: str) -> None:
    for key in keys:
        if key not in obj:
            fail(errors, sequence_id, f"missing {prefix}{key}")


def validate_record(record: dict[str, Any], line_no: int) -> list[str]:
    sequence_id = str(record.get("sequence_id", f"line_{line_no}"))
    errors: list[str] = []

    require_keys(errors, sequence_id, record, REQUIRED_TOP_LEVEL, "")
    if errors:
        return errors

    if record["schema_version"] != "0.1.0":
        fail(errors, sequence_id, "schema_version must be 0.1.0")
    if not str(record["sequence_id"]).startswith("sdsa_"):
        fail(errors, sequence_id, "sequence_id must start with sdsa_")

    dsa = record["dsa_frame_sequence"]
    mask = record["vessel_mask_sequence"]
    require_keys(errors, sequence_id, dsa, ["uri", "frame_count", "height", "width", "dtype"], "dsa_frame_sequence.")
    require_keys(errors, sequence_id, mask, ["uri", "frame_count", "height", "width", "dtype"], "vessel_mask_sequence.")
    if errors:
        return errors

    frame_count = dsa["frame_count"]
    if frame_count <= 0:
        fail(errors, sequence_id, "frame_count must be positive")
    if mask["frame_count"] != frame_count:
        fail(errors, sequence_id, "mask frame_count must equal DSA frame_count")
    if (mask["height"], mask["width"]) != (dsa["height"], dsa["width"]):
        fail(errors, sequence_id, "mask dimensions must equal DSA dimensions")

    bolus = record["bolus_curve"]
    phases = bolus.get("phase_by_frame", [])
    if len(phases) != frame_count:
        fail(errors, sequence_id, "bolus phase_by_frame length must equal frame_count")
    if not set(phases).issubset(VALID_PHASES):
        fail(errors, sequence_id, "bolus phases include unknown values")
    if not (0 <= bolus.get("arrival_frame", -1) <= bolus.get("peak_frame", -1) <= bolus.get("washout_frame", -1) < frame_count):
        fail(errors, sequence_id, "bolus frames must satisfy arrival <= peak <= washout < frame_count")

    tip = record["catheter_tip_state"]
    tip_xy = tip.get("tip_xy_by_frame", [])
    visibility = tip.get("visibility_by_frame", [])
    confidence = tip.get("confidence_target_by_frame", [])
    if len(tip_xy) != frame_count or len(visibility) != frame_count or len(confidence) != frame_count:
        fail(errors, sequence_id, "catheter tip arrays must equal frame_count")
    for idx, xy in enumerate(tip_xy):
        if xy is None:
            continue
        if len(xy) != 2:
            fail(errors, sequence_id, f"tip_xy_by_frame[{idx}] must be [x, y] or null")
            continue
        x, y = xy
        if not (0 <= x < dsa["width"] and 0 <= y < dsa["height"]):
            fail(errors, sequence_id, f"tip_xy_by_frame[{idx}] outside frame bounds")
    for idx, c in enumerate(confidence):
        if not (0.0 <= c <= 1.0):
            fail(errors, sequence_id, f"confidence_target_by_frame[{idx}] outside [0, 1]")

    relationship = record["device_vessel_relationship"].get("state_by_frame", [])
    if len(relationship) != frame_count:
        fail(errors, sequence_id, "device_vessel_relationship.state_by_frame length must equal frame_count")

    catheter = record["catheter_path"]
    if len(catheter.get("occlusion_flags_by_frame", [])) != frame_count:
        fail(errors, sequence_id, "catheter_path.occlusion_flags_by_frame length must equal frame_count")
    if len(catheter.get("polyline_px", [])) < 2:
        fail(errors, sequence_id, "catheter_path.polyline_px needs at least two points")

    tasks = record.get("benchmark_tasks", [])
    if not tasks or not set(tasks).issubset(VALID_TASKS):
        fail(errors, sequence_id, "benchmark_tasks must contain known task names")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_jsonl", type=Path)
    args = parser.parse_args()

    all_errors: list[str] = []
    count = 0
    with args.manifest_jsonl.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                all_errors.append(f"line_{line_no}: invalid JSON: {exc}")
                continue
            all_errors.extend(validate_record(record, line_no))

    if count == 0:
        all_errors.append("manifest contains zero records")

    if all_errors:
        print("Manifest validation failed:", file=sys.stderr)
        for error in all_errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Validated {count} manifest record(s): {args.manifest_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
