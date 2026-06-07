#!/usr/bin/env python3
"""Tiny learned CPU baseline for synthetic DSA verifier runs.

No torch/sklearn: learns a global pixel-intensity mask threshold, a brightest-pixel
catheter-tip offset, and phase intensity centroids from train sequences, then
reports holdout metrics. This is intentionally weak but genuinely fitted from
labels, so it gives a first learned baseline without GPU spend.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("verifier_metrics", SCRIPT_DIR / "verifier_metrics.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

PHASES = ["precontrast", "arrival", "arterial_peak", "washout"]


def frame_files(record: dict[str, Any], root: Path) -> list[Path]:
    count = int(record["dsa_frame_sequence"]["frame_count"])
    frame_dir = verifier.resolve_uri(root, record["dsa_frame_sequence"]["uri"])
    return verifier.frame_paths(frame_dir, "frame", count)


def mask_files(record: dict[str, Any], root: Path) -> list[Path]:
    count = int(record["vessel_mask_sequence"]["frame_count"])
    mask_dir = verifier.resolve_uri(root, record["vessel_mask_sequence"]["uri"])
    return verifier.frame_paths(mask_dir, "mask", count)


def device_mask_files(record: dict[str, Any], root: Path) -> list[Path]:
    if "device_mask_sequence" not in record:
        return []
    count = int(record["device_mask_sequence"]["frame_count"])
    mask_dir = verifier.resolve_uri(root, record["device_mask_sequence"]["uri"])
    return verifier.frame_paths(mask_dir, "device", count)


def image_intensity_samples(frame_path: Path, mask_path: Path) -> tuple[list[int], list[int]]:
    frame = Image.open(frame_path).convert("L")
    mask = Image.open(mask_path).convert("L")
    fp = frame.load()
    mp = mask.load()
    w, h = frame.size
    fg: list[int] = []
    bg: list[int] = []
    for y in range(h):
        for x in range(w):
            if mp[x, y] > 0:
                fg.append(fp[x, y])
            else:
                bg.append(fp[x, y])
    return fg, bg


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = max(0, min(len(vals) - 1, round((len(vals) - 1) * q)))
    return float(vals[idx])


def train_intensity_model(records: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    fg: list[int] = []
    bg: list[int] = []
    tip_dx: list[float] = []
    tip_dy: list[float] = []
    phase_intensities: dict[str, list[float]] = defaultdict(list)

    for record in records:
        frames = frame_files(record, root)
        masks = mask_files(record, root)
        true_tips = record["catheter_tip_state"]["tip_xy_by_frame"]
        phases = record["bolus_curve"]["phase_by_frame"]
        for idx, (frame_path, mask_path) in enumerate(zip(frames, masks)):
            fgi, bgi = image_intensity_samples(frame_path, mask_path)
            fg.extend(fgi)
            bg.extend(bgi[:: max(1, len(bgi) // 500)])
            bright = verifier.brightest_pixel_xy(frame_path)
            true_tip = true_tips[idx]
            if true_tip is not None:
                tip_dx.append(float(true_tip[0]) - bright[0])
                tip_dy.append(float(true_tip[1]) - bright[1])
            phase_intensities[phases[idx]].append(verifier.mean_intensity(frame_path))

    bg_hi = percentile([float(v) for v in bg], 0.995)
    fg_lo = percentile([float(v) for v in fg], 0.05)
    threshold = int(round((bg_hi + fg_lo) / 2))
    threshold = max(1, min(254, threshold))
    phase_centroids = {}
    for phase in PHASES:
        vals = phase_intensities.get(phase, [])
        phase_centroids[phase] = mean(vals) if vals else None
    fallback = mean([v for v in phase_centroids.values() if v is not None]) if any(v is not None for v in phase_centroids.values()) else 0.0
    phase_centroids = {k: (fallback if v is None else v) for k, v in phase_centroids.items()}
    return {
        "name": "tiny_intensity_cpu_baseline",
        "version": "0.1.0",
        "threshold": threshold,
        "tip_offset_xy": [mean(tip_dx) if tip_dx else 0.0, mean(tip_dy) if tip_dy else 0.0],
        "phase_centroids": phase_centroids,
        "train_sequence_count": len(records),
        "train_frame_count": sum(int(r["dsa_frame_sequence"]["frame_count"]) for r in records),
    }


def device_intensity_samples(frame_path: Path, device_mask_path: Path) -> tuple[list[int], list[int]]:
    frame = Image.open(frame_path).convert("L")
    mask = Image.open(device_mask_path).convert("L")
    fp = frame.load()
    mp = mask.load()
    w, h = frame.size
    pos: list[int] = []
    neg: list[int] = []
    for y in range(h):
        for x in range(w):
            if mp[x, y] > 0:
                pos.append(fp[x, y])
            else:
                neg.append(fp[x, y])
    return pos, neg


def train_heatmap_model(records: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    """Train a dependency-light pixel/heatmap baseline from masks and tip labels.

    The model learns the existing vessel threshold plus a separate catheter/device
    threshold from device masks, and a frame-indexed spatial prior for distal tip
    location. Prediction scores candidate bright/device pixels with intensity,
    endpointness, and distance to the learned temporal tip prior.
    """
    model = train_intensity_model(records, root)
    device_pos: list[int] = []
    device_neg: list[int] = []
    priors: dict[int, list[tuple[float, float]]] = defaultdict(list)
    device_sequence_count = 0

    for record in records:
        frames = frame_files(record, root)
        devices = device_mask_files(record, root)
        if devices:
            device_sequence_count += 1
        tips = record["catheter_tip_state"]["tip_xy_by_frame"]
        frame_count = max(1, len(frames))
        if "width" in record["dsa_frame_sequence"] and "height" in record["dsa_frame_sequence"]:
            width = float(record["dsa_frame_sequence"]["width"] - 1)
            height = float(record["dsa_frame_sequence"]["height"] - 1)
        else:
            with Image.open(frames[0]) as image:
                width = float(image.size[0] - 1)
                height = float(image.size[1] - 1)
        for idx, frame_path in enumerate(frames):
            bucket = round(idx / max(1, frame_count - 1) * 5)
            tip = tips[idx]
            if tip is not None:
                priors[bucket].append((float(tip[0]) / max(1.0, width), float(tip[1]) / max(1.0, height)))
            if idx < len(devices):
                pos, neg = device_intensity_samples(frame_path, devices[idx])
                device_pos.extend(pos)
                device_neg.extend(neg[:: max(1, len(neg) // 500)])

    if device_pos:
        neg_hi = percentile([float(v) for v in device_neg], 0.997)
        pos_lo = percentile([float(v) for v in device_pos], 0.10)
        device_threshold = int(round((neg_hi + pos_lo) / 2))
        device_threshold = max(int(model["threshold"]), min(254, device_threshold))
    else:
        device_threshold = min(254, int(model["threshold"]) + 25)

    spatial = []
    for bucket in range(6):
        pts = priors.get(bucket) or priors.get(max(0, min(5, bucket - 1))) or priors.get(max(0, min(5, bucket + 1))) or [(0.5, 0.5)]
        spatial.append([mean([p[0] for p in pts]), mean([p[1] for p in pts])])

    model.update({
        "name": "tiny_heatmap_cpu_baseline",
        "version": "0.3.0",
        "device_threshold": device_threshold,
        "train_device_mask_sequence_count": device_sequence_count,
        "tip_spatial_prior_by_frame_norm": spatial,
        "tip_prior_weight": 4.0,
        "tip_intensity_weight": 0.01,
        "tip_endpoint_weight": 2.0,
    })
    return model


def normalized_patch_vector(image: Image.Image, xy: list[float] | tuple[float, float], radius: int) -> list[float]:
    pixels = image.load()
    w, h = image.size
    cx, cy = int(round(float(xy[0]))), int(round(float(xy[1])))
    values: list[float] = []
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if 0 <= x < w and 0 <= y < h:
                values.append(float(pixels[x, y]))
            else:
                values.append(0.0)
    hi = max(values) if values else 0.0
    if hi <= 0:
        return [0.0 for _ in values]
    return [v / hi for v in values]


def train_patch_heatmap_model(records: list[dict[str, Any]], root: Path, patch_radius: int = 2) -> dict[str, Any]:
    """Train a local patch-template heatmap scorer for catheter-tip pixels."""
    model = train_heatmap_model(records, root)
    patch_vectors: list[list[float]] = []
    for record in records:
        frames = frame_files(record, root)
        tips = record["catheter_tip_state"]["tip_xy_by_frame"]
        for idx, frame_path in enumerate(frames):
            if idx >= len(tips) or tips[idx] is None:
                continue
            with Image.open(frame_path) as image:
                patch_vectors.append(normalized_patch_vector(image.convert("L"), tips[idx], patch_radius))
    patch_len = (patch_radius * 2 + 1) ** 2
    if patch_vectors:
        template = [mean([vec[i] for vec in patch_vectors]) for i in range(patch_len)]
    else:
        template = [0.0 for _ in range(patch_len)]
    model.update({
        "name": "tiny_patch_heatmap_cpu_baseline",
        "version": "0.5.0",
        "patch_radius": patch_radius,
        "tip_patch_template": template,
        "train_patch_count": len(patch_vectors),
        "tip_patch_weight": 8.0,
        "tip_prior_weight": 1.0,
        "tip_intensity_weight": 0.002,
        "tip_endpoint_weight": 1.0,
        "tip_continuity_weight": 0.05,
        "tip_candidate_source": "device_mask_when_available_else_threshold",
    })
    return model


def train_patch_ranker_model(records: list[dict[str, Any]], root: Path, patch_radius: int = 2, epochs: int = 12, learning_rate: float = 0.12) -> dict[str, Any]:
    """Train a tiny dependency-free logistic patch ranker for tip candidates.

    This is intentionally small: it learns to rank device-mask candidate pixels
    from positive tip labels and within-frame hard negatives. It uses no torch or
    sklearn, so it can run as a local CPU smoke model before GPU spend.
    """
    model = train_patch_heatmap_model(records, root, patch_radius=patch_radius)
    examples: list[tuple[list[float], int]] = []
    positive_count = 0
    negative_count = 0
    feature_names = ["bias"] + [f"patch_{idx}" for idx in range((patch_radius * 2 + 1) ** 2)] + [
        "intensity_norm",
        "endpointness",
        "prior_closeness",
        "continuity_closeness",
    ]

    for record in records:
        frames = frame_files(record, root)
        devices = device_mask_files(record, root)
        tips = record["catheter_tip_state"]["tip_xy_by_frame"]
        previous_truth: list[float] | None = None
        for idx, frame_path in enumerate(frames):
            if idx >= len(tips) or tips[idx] is None:
                continue
            device_path = devices[idx] if idx < len(devices) else None
            threshold = int(model.get("device_threshold", model.get("threshold", 180)))
            candidates, values, size = device_mask_candidate_pixels(frame_path, device_path, threshold)
            if not candidates:
                continue
            truth = [float(tips[idx][0]), float(tips[idx][1])]
            with Image.open(frame_path) as image_raw:
                image = image_raw.convert("L")
                for x, y in candidates:
                    dist = math.hypot(x - truth[0], y - truth[1])
                    if dist <= 1.0:
                        label = 1
                        positive_count += 1
                    elif dist >= 3.0:
                        label = 0
                        negative_count += 1
                    else:
                        continue
                    examples.append((ranker_feature_vector(image, (x, y), candidates, values, size, idx, len(frames), model, previous_truth), label))
            previous_truth = truth

    weights = [0.0 for _ in feature_names]
    if examples:
        for _ in range(max(1, epochs)):
            for features, label in examples:
                z = sum(w * f for w, f in zip(weights, features))
                pred = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                # Positives are rare relative to device-mask negatives.
                sample_weight = 3.0 if label == 1 else 1.0
                err = (float(label) - pred) * sample_weight
                for i, feature in enumerate(features):
                    weights[i] += learning_rate * err * feature

    model.update({
        "name": "tiny_patch_ranker_cpu_baseline",
        "version": "0.1.0",
        "ranker_feature_names": feature_names,
        "ranker_weights": weights,
        "train_ranker_example_count": len(examples),
        "train_ranker_positive_count": positive_count,
        "train_ranker_negative_count": negative_count,
        "ranker_learning_rate": learning_rate,
        "ranker_epochs": epochs,
        "ranker_transition_weight": 0.05,
        "ranker_dp_max_candidates": 80,
    })
    return model


def predict_phase(frame_path: Path, model: dict[str, Any]) -> str:
    val = verifier.mean_intensity(frame_path)
    return min(model["phase_centroids"], key=lambda p: abs(model["phase_centroids"][p] - val))


def temporal_phase_predictions(intensities: list[float]) -> list[str]:
    """Predict bolus phase from within-sequence intensity order.

    This encodes the weak temporal prior for short DSA clips: first frame is
    precontrast, the brightest frame is arterial peak, frames between are arrival,
    and later frames are washout.
    """
    if not intensities:
        return []
    peak_idx = max(range(len(intensities)), key=lambda idx: intensities[idx])
    phases = []
    for idx in range(len(intensities)):
        if idx == 0:
            phases.append("precontrast")
        elif idx < peak_idx:
            phases.append("arrival")
        elif idx == peak_idx:
            phases.append("arterial_peak")
        else:
            phases.append("washout")
    return phases


def bright_pixels(frame_path: Path, threshold: int) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], tuple[int, int]]:
    image = Image.open(frame_path).convert("L")
    pixels = image.load()
    w, h = image.size
    coords: set[tuple[int, int]] = set()
    values: dict[tuple[int, int], int] = {}
    for y in range(h):
        for x in range(w):
            val = pixels[x, y]
            if val >= threshold:
                coords.add((x, y))
                values[(x, y)] = val
    return coords, values, (w, h)


def device_mask_candidate_pixels(frame_path: Path, device_mask_path: Path | None, threshold: int) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], tuple[int, int]]:
    """Return candidate tip pixels, preferring typed device masks when present.

    Brightness-threshold candidates are useful for old manifests without device
    masks. For v2+ manifests, the device mask is the typed artifact that says
    which pixels are catheter/wire; use it even when the rendered device is faint
    and below the learned intensity threshold.
    """
    image = Image.open(frame_path).convert("L")
    fp = image.load()
    w, h = image.size
    if device_mask_path is not None and device_mask_path.exists():
        mask_pixels = verifier.load_binary_mask(device_mask_path)
        if mask_pixels:
            bright_mask_pixels = {xy for xy in mask_pixels if int(fp[xy[0], xy[1]]) >= threshold}
            candidate_pixels = bright_mask_pixels or set(mask_pixels)
            return candidate_pixels, {xy: int(fp[xy[0], xy[1]]) for xy in candidate_pixels}, (w, h)
    return bright_pixels(frame_path, threshold)


def endpointness(pixel: tuple[int, int], candidates: set[tuple[int, int]]) -> float:
    x, y = pixel
    neighbors = 0
    for yy in range(y - 1, y + 2):
        for xx in range(x - 1, x + 2):
            if (xx, yy) == (x, y):
                continue
            if (xx, yy) in candidates:
                neighbors += 1
    return 1.0 / (1.0 + neighbors)


def ranker_feature_vector(
    image: Image.Image,
    pixel: tuple[int, int],
    candidates: set[tuple[int, int]],
    values: dict[tuple[int, int], int],
    size: tuple[int, int],
    frame_index: int,
    frame_count: int,
    model: dict[str, Any],
    previous_tip_xy: list[float] | None,
) -> list[float]:
    """Feature vector for the tiny patch-ranker candidate classifier."""
    x, y = pixel
    w, h = size
    radius = int(model.get("patch_radius", 2))
    patch = normalized_patch_vector(image, [x, y], radius)
    priors = model.get("tip_spatial_prior_by_frame_norm") or [[0.5, 0.5]]
    prior_idx = round(frame_index / max(1, frame_count - 1) * (len(priors) - 1))
    prior = priors[max(0, min(len(priors) - 1, prior_idx))]
    px = float(prior[0]) * max(1, w - 1)
    py = float(prior[1]) * max(1, h - 1)
    diag = math.hypot(max(1, w - 1), max(1, h - 1))
    prior_closeness = 1.0 - min(1.0, math.hypot(x - px, y - py) / max(1.0, diag))
    continuity_closeness = 0.0
    if previous_tip_xy is not None:
        continuity_closeness = 1.0 - min(1.0, math.hypot(x - float(previous_tip_xy[0]), y - float(previous_tip_xy[1])) / max(1.0, diag))
    return [
        1.0,
        *patch,
        float(values.get(pixel, 0)) / 255.0,
        endpointness(pixel, candidates),
        prior_closeness,
        continuity_closeness,
    ]


def predict_tip_heatmap(frame_path: Path, frame_index: int, frame_count: int, model: dict[str, Any], device_mask_path: Path | None = None, previous_tip_xy: list[float] | None = None) -> list[float]:
    threshold = int(model.get("device_threshold", model.get("threshold", 180)))
    candidates, values, size = device_mask_candidate_pixels(frame_path, device_mask_path, threshold)
    if not candidates:
        return verifier.brightest_pixel_xy(frame_path)
    w, h = size
    priors = model.get("tip_spatial_prior_by_frame_norm") or [[0.5, 0.5]]
    prior_idx = round(frame_index / max(1, frame_count - 1) * (len(priors) - 1))
    prior = priors[max(0, min(len(priors) - 1, prior_idx))]
    px = float(prior[0]) * max(1, w - 1)
    py = float(prior[1]) * max(1, h - 1)
    prior_weight = float(model.get("tip_prior_weight", 4.0))
    intensity_weight = float(model.get("tip_intensity_weight", 0.01))
    endpoint_weight = float(model.get("tip_endpoint_weight", 2.0))
    continuity_weight = float(model.get("tip_continuity_weight", 0.0))
    diag = math.hypot(max(1, w - 1), max(1, h - 1))
    best: tuple[float, int, int] | None = None
    for x, y in candidates:
        distance_penalty = math.hypot(x - px, y - py) / max(1.0, diag)
        continuity_penalty = 0.0
        if previous_tip_xy is not None:
            continuity_penalty = math.hypot(x - float(previous_tip_xy[0]), y - float(previous_tip_xy[1])) / max(1.0, diag)
        score = (
            values[(x, y)] * intensity_weight
            + endpointness((x, y), candidates) * endpoint_weight
            - prior_weight * distance_penalty
            - continuity_weight * continuity_penalty
        )
        if best is None or score > best[0]:
            best = (score, x, y)
    assert best is not None
    return [float(best[1]), float(best[2])]


def patch_similarity(vector: list[float], template: list[float]) -> float:
    if not vector or not template or len(vector) != len(template):
        return 0.0
    mse = mean([(a - b) ** 2 for a, b in zip(vector, template)])
    return 1.0 - mse


def predict_tip_patch_heatmap(frame_path: Path, frame_index: int, frame_count: int, model: dict[str, Any], device_mask_path: Path | None = None, previous_tip_xy: list[float] | None = None) -> list[float]:
    threshold = int(model.get("device_threshold", model.get("threshold", 180)))
    candidates, values, size = device_mask_candidate_pixels(frame_path, device_mask_path, threshold)
    if not candidates:
        return verifier.brightest_pixel_xy(frame_path)
    w, h = size
    priors = model.get("tip_spatial_prior_by_frame_norm") or [[0.5, 0.5]]
    prior_idx = round(frame_index / max(1, frame_count - 1) * (len(priors) - 1))
    prior = priors[max(0, min(len(priors) - 1, prior_idx))]
    px = float(prior[0]) * max(1, w - 1)
    py = float(prior[1]) * max(1, h - 1)
    prior_weight = float(model.get("tip_prior_weight", 1.0))
    intensity_weight = float(model.get("tip_intensity_weight", 0.002))
    endpoint_weight = float(model.get("tip_endpoint_weight", 1.0))
    patch_weight = float(model.get("tip_patch_weight", 8.0))
    continuity_weight = float(model.get("tip_continuity_weight", 0.0))
    template = [float(v) for v in model.get("tip_patch_template", [])]
    radius = int(model.get("patch_radius", 2))
    diag = math.hypot(max(1, w - 1), max(1, h - 1))
    with Image.open(frame_path) as image_raw:
        image = image_raw.convert("L")
        best: tuple[float, int, int] | None = None
        for x, y in candidates:
            distance_penalty = math.hypot(x - px, y - py) / max(1.0, diag)
            continuity_penalty = 0.0
            if previous_tip_xy is not None:
                continuity_penalty = math.hypot(x - float(previous_tip_xy[0]), y - float(previous_tip_xy[1])) / max(1.0, diag)
            vec = normalized_patch_vector(image, [x, y], radius)
            score = (
                patch_similarity(vec, template) * patch_weight
                + values[(x, y)] * intensity_weight
                + endpointness((x, y), candidates) * endpoint_weight
                - prior_weight * distance_penalty
                - continuity_weight * continuity_penalty
            )
            if best is None or score > best[0]:
                best = (score, x, y)
    assert best is not None
    return [float(best[1]), float(best[2])]


def ranker_candidate_scores(frame_path: Path, frame_index: int, frame_count: int, model: dict[str, Any], device_mask_path: Path | None = None, previous_tip_xy: list[float] | None = None) -> list[tuple[float, float, float]]:
    threshold = int(model.get("device_threshold", model.get("threshold", 180)))
    candidates, values, size = device_mask_candidate_pixels(frame_path, device_mask_path, threshold)
    if not candidates:
        bright = verifier.brightest_pixel_xy(frame_path)
        return [(float(bright[0]), float(bright[1]), 0.0)]
    weights = [float(v) for v in model.get("ranker_weights", [])]
    scored: list[tuple[float, float, float]] = []
    with Image.open(frame_path) as image_raw:
        image = image_raw.convert("L")
        for x, y in candidates:
            features = ranker_feature_vector(image, (x, y), candidates, values, size, frame_index, frame_count, model, previous_tip_xy)
            if len(weights) == len(features):
                score = sum(w * f for w, f in zip(weights, features))
            else:
                score = patch_similarity(normalized_patch_vector(image, [x, y], int(model.get("patch_radius", 2))), model.get("tip_patch_template", []))
            scored.append((float(x), float(y), float(score)))
    max_candidates = int(model.get("ranker_dp_max_candidates", 80))
    scored.sort(key=lambda row: row[2], reverse=True)
    return scored[:max(1, max_candidates)]


def predict_tip_patch_ranker(frame_path: Path, frame_index: int, frame_count: int, model: dict[str, Any], device_mask_path: Path | None = None, previous_tip_xy: list[float] | None = None) -> list[float]:
    scored = ranker_candidate_scores(frame_path, frame_index, frame_count, model, device_mask_path, previous_tip_xy)
    best = scored[0]
    return [float(best[0]), float(best[1])]


def smooth_candidate_path_dp(candidate_scores_by_frame: list[list[tuple[float, float, float]]], transition_weight: float) -> list[list[float]]:
    """Viterbi-style smoother over framewise tip-candidate scores."""
    if not candidate_scores_by_frame:
        return []
    dp: list[list[float]] = []
    back: list[list[int]] = []
    for frame_idx, candidates in enumerate(candidate_scores_by_frame):
        if not candidates:
            candidates = [(0.0, 0.0, 0.0)]
            candidate_scores_by_frame[frame_idx] = candidates
        if frame_idx == 0:
            dp.append([float(score) for _, _, score in candidates])
            back.append([-1 for _ in candidates])
            continue
        prev_candidates = candidate_scores_by_frame[frame_idx - 1]
        row: list[float] = []
        brow: list[int] = []
        for x, y, score in candidates:
            best_prev_idx = 0
            best_prev_score = None
            for prev_idx, (px, py, _) in enumerate(prev_candidates):
                transition_penalty = math.hypot(float(x) - float(px), float(y) - float(py)) * transition_weight
                total = dp[frame_idx - 1][prev_idx] + float(score) - transition_penalty
                if best_prev_score is None or total > best_prev_score:
                    best_prev_score = total
                    best_prev_idx = prev_idx
            row.append(float(best_prev_score if best_prev_score is not None else score))
            brow.append(best_prev_idx)
        dp.append(row)
        back.append(brow)
    last_idx = max(range(len(dp[-1])), key=lambda idx: dp[-1][idx])
    chosen = [last_idx]
    for frame_idx in range(len(candidate_scores_by_frame) - 1, 0, -1):
        last_idx = back[frame_idx][last_idx]
        chosen.append(last_idx)
    chosen.reverse()
    return [[float(candidate_scores_by_frame[idx][cand_idx][0]), float(candidate_scores_by_frame[idx][cand_idx][1])] for idx, cand_idx in enumerate(chosen)]


def predict_tips_patch_ranker_dp(frames: list[Path], devices: list[Path], model: dict[str, Any]) -> list[list[float]]:
    candidate_scores_by_frame = []
    for idx, frame_path in enumerate(frames):
        device_path = devices[idx] if idx < len(devices) else None
        candidate_scores_by_frame.append(ranker_candidate_scores(frame_path, idx, len(frames), model, device_path, None))
    return smooth_candidate_path_dp(candidate_scores_by_frame, float(model.get("ranker_transition_weight", 1.0)))


def predict_record(record: dict[str, Any], root: Path, model: dict[str, Any]) -> dict[str, Any]:
    frames = frame_files(record, root)
    devices = device_mask_files(record, root)
    threshold = int(model["threshold"])
    name = str(model.get("name", ""))
    if name.startswith("tiny_patch_ranker_dp"):
        tips = predict_tips_patch_ranker_dp(frames, devices, model)
    elif name.startswith("tiny_patch_ranker"):
        tips = []
        previous_tip = None
        for idx, frame_path in enumerate(frames):
            device_path = devices[idx] if idx < len(devices) else None
            tip = predict_tip_patch_ranker(frame_path, idx, len(frames), model, device_path, previous_tip)
            tips.append(tip)
            previous_tip = tip
    elif name.startswith("tiny_patch_heatmap"):
        tips = []
        previous_tip = None
        for idx, frame_path in enumerate(frames):
            device_path = devices[idx] if idx < len(devices) else None
            tip = predict_tip_patch_heatmap(frame_path, idx, len(frames), model, device_path, previous_tip)
            tips.append(tip)
            previous_tip = tip
    elif name.startswith("tiny_heatmap"):
        tips = []
        previous_tip = None
        for idx, frame_path in enumerate(frames):
            device_path = devices[idx] if idx < len(devices) else None
            tip = predict_tip_heatmap(frame_path, idx, len(frames), model, device_path, previous_tip)
            tips.append(tip)
            previous_tip = tip
    else:
        ox, oy = model["tip_offset_xy"]
        tips = []
        for frame_path in frames:
            bright = verifier.brightest_pixel_xy(frame_path)
            tips.append([bright[0] + ox, bright[1] + oy])
    phase_rule = model.get("phase_rule", "centroid")
    if phase_rule == "temporal_rank":
        phase_by_frame = temporal_phase_predictions([verifier.mean_intensity(path) for path in frames])
    elif phase_rule == "centroid":
        phase_by_frame = [predict_phase(path, model) for path in frames]
    else:
        raise ValueError("phase_rule must be centroid or temporal_rank")
    return {
        "mask_pixels_by_frame": [verifier.threshold_mask_pixels(path, threshold=threshold) for path in frames],
        "tip_xy_by_frame": tips,
        "phase_by_frame": phase_by_frame,
    }


def evaluate_record(record: dict[str, Any], root: Path, model: dict[str, Any]) -> dict[str, Any]:
    sequence_id = record["sequence_id"]
    truth_masks = mask_files(record, root)
    pred = predict_record(record, root, model)
    true_tips = record["catheter_tip_state"]["tip_xy_by_frame"]
    occlusion_flags = record.get("catheter_path", {}).get("occlusion_flags_by_frame", [False] * len(truth_masks))
    per_frame = []
    ious = []
    dices = []
    tip_errors = []
    threshold_hits = {t: [] for t in verifier.DEFAULT_THRESHOLDS}
    for idx, truth_mask_path in enumerate(truth_masks):
        mask_metrics = verifier.binary_mask_metrics_from_pixels(verifier.load_binary_mask(truth_mask_path), pred["mask_pixels_by_frame"][idx])
        tip_metrics = verifier.tip_localization_metrics(true_tips[idx], pred["tip_xy_by_frame"][idx])
        ious.append(float(mask_metrics["iou"]))
        dices.append(float(mask_metrics["dice"]))
        if tip_metrics["tip_error_px"] is not None:
            tip_errors.append(float(tip_metrics["tip_error_px"]))
        for t in verifier.DEFAULT_THRESHOLDS:
            threshold_hits[t].append(bool(tip_metrics[f"within_{t}px"]))
        per_frame.append({
            "frame_index": idx,
            "occluded": bool(occlusion_flags[idx]) if idx < len(occlusion_flags) else False,
            "predicted_tip_xy": pred["tip_xy_by_frame"][idx],
            **mask_metrics,
            **tip_metrics,
        })
    phase = verifier.bolus_phase_metrics(record["bolus_curve"]["phase_by_frame"], pred["phase_by_frame"])
    summary = {
        "sequence_id": sequence_id,
        "baseline": model["name"],
        "frame_count": len(truth_masks),
        "failure_modes": record.get("failure_modes", []),
        "view": record.get("projection_view", {}).get("view"),
        "overlap_score": record.get("projection_view", {}).get("overlap_score"),
        "mean_iou": mean(ious) if ious else 0.0,
        "min_iou": min(ious) if ious else 0.0,
        "mean_dice": mean(dices) if dices else 0.0,
        "min_dice": min(dices) if dices else 0.0,
        "mean_tip_error_px": mean(tip_errors) if tip_errors else None,
        "max_tip_error_px": max(tip_errors) if tip_errors else None,
        "tip_occlusion_rate": sum(1 for v in occlusion_flags if v) / len(truth_masks) if truth_masks else 0.0,
        "phase_accuracy": phase["phase_accuracy"],
        "phase_mae_frames": phase["phase_mae_frames"],
        "phase_confusion": phase["phase_confusion"],
        "per_frame": per_frame,
    }
    for t, hits in threshold_hits.items():
        summary[f"tip_within_{t}px_rate"] = sum(hits) / len(hits) if hits else 0.0
    return summary


def aggregate(summaries: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        vals = [s[key] for s in summaries if s.get(key) is not None]
        return mean(vals) if vals else None

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        modes = summary.get("failure_modes") or ["none"]
        for mode in modes:
            groups[mode].append(summary)
    return {
        "model": model,
        "sequence_count": len(summaries),
        "aggregate": {
            "mean_iou": avg("mean_iou"),
            "min_iou": min([s["min_iou"] for s in summaries], default=None),
            "mean_dice": avg("mean_dice"),
            "mean_tip_error_px": avg("mean_tip_error_px"),
            "max_tip_error_px": max([s["max_tip_error_px"] for s in summaries if s["max_tip_error_px"] is not None], default=None),
            "tip_within_2px_rate": avg("tip_within_2px_rate"),
            "tip_within_5px_rate": avg("tip_within_5px_rate"),
            "tip_within_10px_rate": avg("tip_within_10px_rate"),
            "phase_accuracy": avg("phase_accuracy"),
            "phase_mae_frames": avg("phase_mae_frames"),
        },
        "by_failure_mode": {
            mode: {
                "sequence_count": len(rows),
                "mean_iou": mean([r["mean_iou"] for r in rows]),
                "mean_tip_error_px": mean([r["mean_tip_error_px"] for r in rows if r["mean_tip_error_px"] is not None]),
                "phase_accuracy": mean([r["phase_accuracy"] for r in rows]),
            }
            for mode, rows in groups.items()
        },
        "sequences": summaries,
    }


def evaluate_records(records: list[dict[str, Any]], root: Path, model: dict[str, Any]) -> dict[str, Any]:
    return aggregate([evaluate_record(record, root, model) for record in records], model)


def build_train_eval_sets(eval_manifest: Path, train_manifests: list[Path] | None, train_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load train/eval records for within-manifest or cross-manifest runs."""
    eval_records_all = verifier.load_manifest(eval_manifest)
    if train_manifests:
        train_records: list[dict[str, Any]] = []
        for manifest in train_manifests:
            train_records.extend(verifier.load_manifest(manifest))
        split = {
            "mode": "cross_manifest",
            "train_count": len(train_records),
            "eval_count": len(eval_records_all),
            "train_fraction": None,
            "train_manifests": [str(p) for p in train_manifests],
            "eval_manifest": str(eval_manifest),
        }
        return train_records, eval_records_all, split

    split_idx = max(1, min(len(eval_records_all) - 1, round(len(eval_records_all) * train_fraction))) if len(eval_records_all) > 1 else len(eval_records_all)
    train_records = eval_records_all[:split_idx]
    eval_records = eval_records_all[split_idx:] if len(eval_records_all) > 1 else eval_records_all
    split = {
        "mode": "within_manifest",
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "train_fraction": train_fraction,
        "train_manifests": [str(eval_manifest)],
        "eval_manifest": str(eval_manifest),
    }
    return train_records, eval_records, split


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Tiny CPU learned baseline report",
        "",
        f"Model: `{report['model']['name']}` v{report['model']['version']}",
        f"Train sequences: {report['model']['train_sequence_count']}",
        f"Eval sequences: {report['sequence_count']}",
        f"Learned threshold: {report['model']['threshold']}",
        f"Learned device threshold: {report['model'].get('device_threshold', 'n/a')}",
        f"Learned tip offset xy: {[round(v, 3) for v in report['model'].get('tip_offset_xy', [])] or 'n/a'}",
        "",
        "## Aggregate metrics",
        "",
    ]
    for key, value in report["aggregate"].items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Failure-mode slices", ""])
    for mode, row in sorted(report["by_failure_mode"].items()):
        lines.append(f"- {mode}")
        for key, value in row.items():
            lines.append(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_jsonl", type=Path, help="Evaluation manifest, or train/eval manifest for within-manifest split when --train-manifest is omitted")
    ap.add_argument("--root", type=Path, default=Path("research/synthetic_dsa"))
    ap.add_argument("--train-fraction", type=float, default=0.7)
    ap.add_argument("--train-manifest", type=Path, action="append", default=[], help="Training manifest for cross-realism runs. Repeat for mixed-regime training.")
    ap.add_argument("--out-json", type=Path, default=Path("research/synthetic_dsa/outputs/reports/tiny_cpu_baseline_report.json"))
    ap.add_argument("--out-md", type=Path, default=Path("research/synthetic_dsa/outputs/reports/tiny_cpu_baseline_report.md"))
    ap.add_argument("--phase-rule", choices=["centroid", "temporal_rank"], default="centroid")
    ap.add_argument("--baseline", choices=["intensity", "heatmap", "patch_heatmap", "patch_ranker", "patch_ranker_dp"], default="intensity", help="Use heatmap for the device-threshold + spatial-prior tip baseline, patch_heatmap for a local patch-template tip regressor, patch_ranker for a tiny trained candidate ranker, or patch_ranker_dp for sequence-smoothed candidate ranking.")
    args = ap.parse_args()

    train_records, eval_records, split = build_train_eval_sets(args.manifest_jsonl, args.train_manifest, args.train_fraction)
    if args.baseline in {"patch_ranker", "patch_ranker_dp"}:
        model = train_patch_ranker_model(train_records, args.root)
    elif args.baseline == "patch_heatmap":
        model = train_patch_heatmap_model(train_records, args.root)
    elif args.baseline == "heatmap":
        model = train_heatmap_model(train_records, args.root)
    else:
        model = train_intensity_model(train_records, args.root)
    model["phase_rule"] = args.phase_rule
    if args.phase_rule == "temporal_rank" and args.baseline == "intensity":
        model["name"] = "tiny_temporal_cpu_baseline"
        model["version"] = "0.2.0"
    elif args.phase_rule == "temporal_rank" and args.baseline == "heatmap":
        model["name"] = "tiny_heatmap_temporal_cpu_baseline"
        model["version"] = "0.3.0"
    elif args.phase_rule == "temporal_rank" and args.baseline == "patch_heatmap":
        model["name"] = "tiny_patch_heatmap_temporal_cpu_baseline"
        model["version"] = "0.5.0"
    elif args.phase_rule == "temporal_rank" and args.baseline == "patch_ranker":
        model["name"] = "tiny_patch_ranker_temporal_cpu_baseline"
        model["version"] = "0.1.0"
    elif args.phase_rule == "temporal_rank" and args.baseline == "patch_ranker_dp":
        model["name"] = "tiny_patch_ranker_dp_temporal_cpu_baseline"
        model["version"] = "0.2.0"
    report = evaluate_records(eval_records, args.root, model)
    report["manifest"] = str(args.manifest_jsonl)
    report["split"] = split
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, args.out_md)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "model": model, "aggregate": report["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
