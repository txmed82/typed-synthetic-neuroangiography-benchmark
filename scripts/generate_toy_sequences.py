#!/usr/bin/env python3
"""CPU-only toy synthetic DSA sequence generator.

Creates DSA-like 2D+t PNG frames, vessel masks, catheter overlays, and manifest
records. v0 is a simple smoke target; v1 adds modest procedural realism while
remaining fully labeled and dependency-light.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter

PHASES = ["precontrast", "arrival", "arrival", "arterial_peak", "washout", "washout"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def bezier(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float], n: int) -> list[tuple[float, float]]:
    pts = []
    for i in range(n):
        t = i / max(1, n - 1)
        x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
        y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def jittered_parent(rng: random.Random, size: int) -> list[tuple[float, float]]:
    p0 = (rng.uniform(14, 24), rng.uniform(size * 0.68, size * 0.82))
    p3 = (rng.uniform(size * 0.72, size * 0.86), rng.uniform(size * 0.25, size * 0.42))
    p1 = (rng.uniform(size * 0.22, size * 0.38), rng.uniform(size * 0.52, size * 0.78))
    p2 = (rng.uniform(size * 0.45, size * 0.68), rng.uniform(size * 0.28, size * 0.55))
    return bezier(p0, p1, p2, p3, 70)


def branch_from(parent: list[tuple[float, float]], rng: random.Random, size: int, side: int, short: bool = False) -> list[tuple[float, float]]:
    idx = rng.randint(20, max(22, len(parent) - 18))
    p0 = parent[idx]
    length = rng.uniform(14, 28) if short else rng.uniform(24, 48)
    p3 = (max(6, min(size - 6, p0[0] + side * length)), max(6, min(size - 6, p0[1] - rng.uniform(16, 46))))
    p1 = (p0[0] + side * rng.uniform(8, 18), p0[1] - rng.uniform(4, 14))
    p2 = (p3[0] - side * rng.uniform(6, 18), p3[1] + rng.uniform(4, 18))
    return bezier(p0, p1, p2, p3, 24 if short else 38)


def path_length(points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def draw_polyline(draw: ImageDraw.ImageDraw, points: Iterable[tuple[float, float]], width: int, fill: int) -> None:
    pts = [(round(x), round(y)) for x, y in points]
    if len(pts) >= 2:
        draw.line(pts, fill=fill, width=width, joint="curve")


def shifted(points: list[tuple[float, float]], dx: float, dy: float) -> list[tuple[float, float]]:
    return [(x + dx, y + dy) for x, y in points]


def point_along(points: list[tuple[float, float]], frac: float) -> tuple[float, float]:
    if frac <= 0:
        return points[0]
    if frac >= 1:
        return points[-1]
    total = path_length(points)
    target = total * frac
    acc = 0.0
    for a, b in zip(points, points[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if acc + seg >= target:
            t = (target - acc) / max(seg, 1e-6)
            return (lerp(a[0], b[0], t), lerp(a[1], b[1], t))
        acc += seg
    return points[-1]


def textured_background(rng: random.Random, size: int, realism: str) -> Image.Image:
    base = Image.new("L", (size, size), rng.randint(5, 12))
    draw = ImageDraw.Draw(base)
    noise_points = 260 if realism == "v0" else 1700 if realism == "v4" else 1300 if realism == "v3" else 900
    for _ in range(noise_points):
        x = rng.randrange(size)
        y = rng.randrange(size)
        val = rng.randrange(0, 9 if realism == "v0" else 36 if realism == "v4" else 30 if realism == "v3" else 22)
        draw.point((x, y), fill=val)
    if realism in {"v1", "v2", "v3", "v4"}:
        # broad subtraction gradients and detector shading
        for y in range(size):
            shade_amp = 5 if realism == "v1" else 15 if realism == "v4" else 12 if realism == "v3" else 9
            ramp_amp = 5 if realism == "v1" else 16 if realism == "v4" else 13 if realism == "v3" else 10
            shade = int(shade_amp * math.sin(y / max(8, size / 10)) + ramp_amp * y / size)
            for x in range(0, size, 3):
                density = 0.45 if realism == "v1" else 0.92 if realism == "v4" else 0.88 if realism == "v3" else 0.75
                if rng.random() < density:
                    draw.point((x, y), fill=max(0, min(255, base.getpixel((x, y)) + shade)))
        for _ in range(4 if realism == "v1" else 17 if realism == "v4" else 13 if realism == "v3" else 9):
            cx, cy = rng.randrange(size), rng.randrange(size)
            radius = rng.uniform(size * 0.18, size * 0.42)
            fill = rng.randrange(4, 16 if realism == "v1" else 38 if realism == "v4" else 32 if realism == "v3" else 26)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=fill)
        if realism in {"v2", "v3", "v4"}:
            for _ in range(42 if realism == "v4" else 32 if realism == "v3" else 18):
                x0 = rng.randrange(size)
                y0 = rng.randrange(size)
                delta = 52 if realism == "v4" else 44 if realism == "v3" else 30
                x1 = max(0, min(size - 1, x0 + rng.randint(-delta, delta)))
                y1 = max(0, min(size - 1, y0 + rng.randint(-delta, delta)))
                draw.line((x0, y0, x1, y1), fill=rng.randrange(5, 34 if realism == "v4" else 28 if realism == "v3" else 22), width=1)
        if realism in {"v3", "v4"}:
            # Faint anatomy/detector shadows and scan-line banding. These make
            # intensity-only segmentation less trivial without changing labels.
            for _ in range(5 if realism == "v4" else 3):
                cx, cy = rng.randrange(size), rng.randrange(size)
                rx, ry = rng.uniform(size * 0.18, size * 0.45), rng.uniform(size * 0.05, size * 0.16)
                draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=rng.randrange(8, 24), width=2)
            for x in range(0, size, 16):
                if rng.random() < 0.45:
                    draw.line((x, 0, x + rng.randint(-4, 4), size - 1), fill=rng.randrange(3, 16), width=1)
        base = base.filter(ImageFilter.GaussianBlur(radius=0.55 if realism == "v1" else 1.15 if realism == "v4" else 1.05 if realism == "v3" else 0.9))
    return base


def make_phases(frames: int) -> list[str]:
    phases = PHASES[:frames]
    while len(phases) < frames:
        phases.append("washout")
    return phases


def make_sequence(seq_idx: int, seed: int, out_dir: Path, size: int, frames: int, realism: str = "v0", tag: str = "toy") -> dict:
    rng = random.Random(seed)
    sequence_id = f"sdsa_{tag}_{seq_idx:04d}"
    seq_dir = out_dir / "sequences" / sequence_id
    frame_dir = seq_dir / "frames"
    mask_dir = seq_dir / "masks"
    device_dir = seq_dir / "device_masks"
    frame_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if realism in {"v2", "v3", "v4"}:
        device_dir.mkdir(parents=True, exist_ok=True)

    parent = jittered_parent(rng, size)
    vessels = [parent, branch_from(parent, rng, size, -1), branch_from(parent, rng, size, 1)]
    if realism in {"v1", "v2", "v3", "v4"}:
        branch_sides = [-1, 1, -1, 1] if realism == "v1" else [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1] if realism == "v4" else [-1, 1, -1, 1, -1, 1, -1, 1] if realism == "v3" else [-1, 1, -1, 1, -1, 1]
        for side in branch_sides:
            vessels.append(branch_from(parent, rng, size, side, short=True))
        if realism in {"v2", "v3", "v4"}:
            # Add a crude aneurysm/sac-like loop as a side-branch primitive.
            sac_anchor = parent[rng.randint(30, 48)]
            sac_radius = rng.uniform(4.0, 7.5)
            sac = [(sac_anchor[0] + math.cos(t / 18 * 2 * math.pi) * sac_radius, sac_anchor[1] + math.sin(t / 18 * 2 * math.pi) * sac_radius) for t in range(19)]
            vessels.append(sac)
        if rng.random() < (0.35 if realism == "v1" else 0.08 if realism == "v4" else 0.12 if realism == "v3" else 0.18):
            failure_modes_seed = ["vessel_sparsity"]
            vessels = [parent] + rng.sample(vessels[1:], k=max(2, len(vessels) - 3))
        else:
            failure_modes_seed = []
    else:
        failure_modes_seed = []

    diameter = rng.randint(3, 6)
    view = "oblique" if realism == "v4" else rng.choice(["AP", "lateral", "oblique"])
    overlap_score = round(rng.uniform(0.76, 0.94) if realism == "v4" else rng.uniform(0.05, 0.86 if realism in {"v1", "v2", "v3"} and view == "oblique" else 0.75 if view == "oblique" else 0.45), 3)
    failure_modes = list(failure_modes_seed)
    if overlap_score > 0.55:
        failure_modes.extend(["overlap", "projection_ambiguity"])
    if rng.random() < (0.45 if realism == "v1" else 0.40 if realism == "v3" else 0.35):
        failure_modes.append("low_contrast")
    if rng.random() < (0.45 if realism == "v1" else 0.38 if realism == "v3" else 0.25):
        failure_modes.append("noise")
    if realism in {"v1", "v2", "v3", "v4"} and rng.random() < (0.45 if realism == "v1" else 0.65 if realism == "v4" else 0.50 if realism == "v3" else 0.35):
        failure_modes.append("motion")
    if realism in {"v3", "v4"}:
        failure_modes.append("coil_mass")

    catheter_type = rng.choice(["centerline_following", "near_wall", "branch_entering", "ambiguous_overlap"])
    catheter_path = parent[:]
    if catheter_type == "branch_entering":
        catheter_path = parent[:35] + vessels[min(2, len(vessels) - 1)][:]
    elif catheter_type == "near_wall":
        catheter_path = [(x + rng.uniform(1.5, 3.5), y + rng.uniform(0.5, 2.5)) for x, y in parent]
    elif catheter_type == "ambiguous_overlap":
        catheter_path = parent[:]
        if "projection_ambiguity" not in failure_modes:
            failure_modes.append("projection_ambiguity")

    phases = make_phases(frames)
    phase_gain = {"precontrast": 0.04, "arrival": 0.42, "arterial_peak": 0.92, "washout": 0.56}
    motion_shift_px_by_frame = []
    if realism in {"v1", "v2", "v3", "v4"}:
        drift_x = rng.uniform(-1.8, 1.8) if realism == "v1" else rng.uniform(-2.2, 2.2) if realism == "v4" else rng.uniform(-1.5, 1.5) if realism == "v3" else rng.uniform(-1.2, 1.2)
        drift_y = rng.uniform(-1.8, 1.8) if realism == "v1" else rng.uniform(-2.2, 2.2) if realism == "v4" else rng.uniform(-1.5, 1.5) if realism == "v3" else rng.uniform(-1.2, 1.2)
        for f in range(frames):
            t = f / max(1, frames - 1)
            motion_shift_px_by_frame.append([round(drift_x * t + rng.uniform(-0.35, 0.35), 2), round(drift_y * t + rng.uniform(-0.35, 0.35), 2)])
    else:
        motion_shift_px_by_frame = [[0.0, 0.0] for _ in range(frames)]

    tip_xy_by_frame = []
    visibility = []
    confidence = []
    relationship = []
    occlusion_flags = []
    noise_sigma = round(rng.uniform(0.04, 0.14) if realism == "v1" else rng.uniform(0.045, 0.105) if realism == "v4" else rng.uniform(0.025, 0.075) if realism == "v3" else rng.uniform(0.01, 0.04), 3)
    blur_radius = round(rng.uniform(0.35, 1.05) if realism == "v1" else rng.uniform(0.75, 1.35) if realism == "v4" else rng.uniform(0.55, 1.15) if realism == "v3" else 0.45, 3)
    subtraction_residual_strength = round(rng.uniform(0.05, 0.22) if realism == "v1" else rng.uniform(0.08, 0.18) if realism == "v4" else rng.uniform(0.035, 0.12) if realism == "v3" else 0.02, 3)
    detector_artifacts = {"bone_shadow_count": 5 if realism == "v4" else 3 if realism == "v3" else 0, "scanline_stride_px": 12 if realism == "v4" else 16 if realism == "v3" else 0}
    stress_protocol = {"name": "coil_projection_ambiguity_v0", "coil_decoy_count": 4, "projection_ambiguity_score": max(0.75, overlap_score), "catheter_salience": 0.62} if realism == "v4" else None

    for f in range(frames):
        phase = phases[f]
        gain = phase_gain[phase]
        if "low_contrast" in failure_modes:
            gain *= 0.62 if realism == "v1" else 0.64 if realism == "v4" else 0.72 if realism == "v3" else 0.68
        bg = textured_background(rng, size, realism)
        mask = Image.new("1", (size, size), 0)
        device_mask = Image.new("1", (size, size), 0)
        draw = ImageDraw.Draw(bg)
        mdraw = ImageDraw.Draw(mask)
        ddraw = ImageDraw.Draw(device_mask)
        dx, dy = motion_shift_px_by_frame[f]
        visible_vessels = vessels if f >= (1 if realism in {"v3", "v4"} else 2) else [parent]
        for idx, pts in enumerate(visible_vessels):
            width = max(1, diameter - (1 if idx > 2 else 0) + (1 if realism in {"v2", "v3", "v4"} and idx == 0 else 0) - (1 if realism in {"v3", "v4"} and idx > 4 else 0))
            intensity = max(0, min(245, int((205 - idx * (4 if realism == "v4" else 5 if realism == "v3" else 6 if realism == "v2" else 9)) * gain)))
            pts_shifted = shifted(pts, dx, dy)
            draw_polyline(draw, pts_shifted, width, intensity)
            draw_polyline(mdraw, pts_shifted, width, 1)
        if realism in {"v1", "v3", "v4"} and subtraction_residual_strength > 0:
            for pts in rng.sample(visible_vessels, k=min(len(visible_vessels), 2)):
                draw_polyline(draw, shifted(pts, dx + rng.uniform(-1.2, 1.2), dy + rng.uniform(-1.2, 1.2)), max(1, diameter - 2), int(80 * subtraction_residual_strength))
        if f > 0:
            bg = bg.filter(ImageFilter.GaussianBlur(radius=blur_radius if "motion" in failure_modes else blur_radius * 0.55))
        tip_frac = 0.20 + 0.68 * (f / max(1, frames - 1))
        tip = point_along(catheter_path, tip_frac)
        tip = (tip[0] + dx, tip[1] + dy)
        tip_xy_by_frame.append([round(tip[0], 2), round(tip[1], 2)])
        occ = catheter_type == "ambiguous_overlap" and f in {frames // 2, frames // 2 + 1}
        if occ and "catheter_occlusion" not in failure_modes:
            failure_modes.append("catheter_occlusion")
        occlusion_flags.append(bool(occ))
        visibility.append("ambiguous" if occ else "visible")
        confidence.append(0.38 if occ and realism == "v1" else 0.45 if occ else 0.95)
        relationship.append({
            "centerline_following": "in_vessel",
            "near_wall": "near_wall",
            "branch_entering": "branch_entering",
            "ambiguous_overlap": "ambiguous" if occ else "crossing_overlap",
        }[catheter_type])
        cdraw = ImageDraw.Draw(bg)
        partial = [point_along(catheter_path, 0.05 + tip_frac * t / 20) for t in range(21)]
        partial = shifted(partial, dx, dy)
        catheter_width = 3 if realism in {"v2", "v3", "v4"} else 2
        catheter_fill = rng.randint(135, 182) if realism == "v4" else rng.randint(155, 205) if realism == "v3" else rng.randint(175, 215) if realism == "v2" else 230 if realism == "v1" else 235
        draw_polyline(cdraw, partial, catheter_width, catheter_fill)
        if realism in {"v2", "v3", "v4"}:
            draw_polyline(ddraw, partial, catheter_width + 1, 1)
        if not occ:
            tip_radius = 1.7 if realism == "v4" else 2.0 if realism == "v3" else 2.2 if realism == "v2" else 2
            tip_fill = min(245, catheter_fill + (9 if realism == "v4" else 14 if realism == "v3" else 20 if realism == "v2" else 25))
            cdraw.ellipse((tip[0] - tip_radius, tip[1] - tip_radius, tip[0] + tip_radius, tip[1] + tip_radius), fill=tip_fill)
            if realism in {"v2", "v3", "v4"}:
                ddraw.ellipse((tip[0] - tip_radius, tip[1] - tip_radius, tip[0] + tip_radius, tip[1] + tip_radius), fill=1)
        else:
            occ_radius = 1.5 if realism in {"v2", "v3", "v4"} else 1
            cdraw.ellipse((tip[0] - occ_radius, tip[1] - occ_radius, tip[0] + occ_radius, tip[1] + occ_radius), fill=142 if realism == "v4" else 150 if realism == "v3" else 165 if realism == "v2" else 185)
            if realism in {"v2", "v3", "v4"}:
                ddraw.ellipse((tip[0] - occ_radius, tip[1] - occ_radius, tip[0] + occ_radius, tip[1] + occ_radius), fill=1)
        if realism in {"v3", "v4"}:
            # Add a small coil/microdevice-like radio-opaque blob near the
            # distal path into pixels only; catheter mask remains separate.
            coil = point_along(catheter_path, min(0.94, tip_frac + 0.04))
            cdraw.ellipse((coil[0] + dx - 2.4, coil[1] + dy - 2.4, coil[0] + dx + 2.4, coil[1] + dy + 2.4), outline=210, width=1)
            if realism == "v4":
                for offset_idx in range(4):
                    frac = min(0.96, tip_frac + 0.06 + offset_idx * 0.025)
                    decoy = point_along(catheter_path, frac)
                    ox = dx + rng.uniform(-5.0, 5.0)
                    oy = dy + rng.uniform(-5.0, 5.0)
                    cdraw.ellipse((decoy[0] + ox - 2.8, decoy[1] + oy - 2.8, decoy[0] + ox + 2.8, decoy[1] + oy + 2.8), outline=rng.randint(190, 230), width=1)
        bg.save(frame_dir / f"frame_{f:03d}.png")
        mask.convert("L").save(mask_dir / f"mask_{f:03d}.png")
        if realism in {"v2", "v3", "v4"}:
            device_mask.convert("L").save(device_dir / f"device_{f:03d}.png")

    montage = Image.new("L", (size * frames, size), 0)
    for f in range(frames):
        montage.paste(Image.open(frame_dir / f"frame_{f:03d}.png"), (f * size, 0))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    montage.save(fig_dir / f"{sequence_id}_montage.png")

    config_hash = hashlib.sha256(json.dumps({"seed": seed, "size": size, "frames": frames, "realism": realism, "tag": tag}, sort_keys=True).encode()).hexdigest()
    branch_labels = ["parent"] + [f"branch_{i}" for i in range(1, len(vessels))]
    version = "0.5.0" if realism == "v4" else "0.4.0" if realism == "v3" else "0.3.0" if realism == "v2" else "0.2.0" if realism == "v1" else "0.1.0"
    catheter_polyline_step = 1 if realism in {"v2", "v3", "v4"} else 10
    branch_diameter_px = [float(max(1, diameter - (0 if i == 0 else 1 if i <= 4 else 2))) for i in range(len(vessels))]
    record = {
        "schema_version": "0.1.0",
        "sequence_id": sequence_id,
        "generator": {"name": "toy_pil_dsa_generator", "version": version, "config_hash": f"sha256:{config_hash}", "seed": seed},
        "provenance": {"created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(), "source": "synthetic"},
        "vascular_graph": {
            "node_count": 2 + len(vessels) * 2,
            "edge_count": max(1, len(vessels) + 2),
            "branch_labels": branch_labels,
            "diameter_px_range": [float(max(1, diameter - 2 if realism == "v1" else diameter - 1)), float(diameter + 1)],
            "tortuosity_mean": round(path_length(parent) / math.hypot(parent[-1][0] - parent[0][0], parent[-1][1] - parent[0][1]), 3),
        },
        "projection_view": {"view": view, "c_arm_degrees": [0.0 if view == "AP" else 90.0 if view == "lateral" else 35.0, 0.0], "overlap_score": overlap_score, "foreshortening_score": round(rng.uniform(0.05, 0.55 if realism == "v1" else 0.45), 3)},
        "bolus_curve": {"arrival_frame": 1, "peak_frame": min(3, frames - 1), "washout_frame": frames - 1, "phase_by_frame": phases},
        "dsa_frame_sequence": {"uri": f"outputs/sequences/{sequence_id}/frames", "frame_count": frames, "height": size, "width": size, "dtype": "uint8"},
        "vessel_mask_sequence": {"uri": f"outputs/sequences/{sequence_id}/masks", "frame_count": frames, "height": size, "width": size, "dtype": "bool"},
        "catheter_path": {"path_type": catheter_type, "polyline_px": [[round(x, 2), round(y, 2)] for x, y in catheter_path[::catheter_polyline_step]], "occlusion_flags_by_frame": occlusion_flags},
        "catheter_tip_state": {"tip_xy_by_frame": tip_xy_by_frame, "visibility_by_frame": visibility, "confidence_target_by_frame": confidence},
        "device_vessel_relationship": {"state_by_frame": relationship},
        "appearance_model": {"realism": realism, "noise_sigma": noise_sigma, "blur_radius": blur_radius, "subtraction_residual_strength": subtraction_residual_strength, "motion_shift_px_by_frame": motion_shift_px_by_frame},
        "failure_modes": sorted(set(failure_modes)),
        "benchmark_tasks": ["vessel_segmentation", "catheter_tip_localization", "bolus_phase_estimation", "failure_detection"],
    }
    if realism in {"v2", "v3", "v4"}:
        record["device_mask_sequence"] = {"uri": f"outputs/sequences/{sequence_id}/device_masks", "frame_count": frames, "height": size, "width": size, "dtype": "bool"}
    if realism in {"v3", "v4"}:
        record["vascular_graph"]["branch_diameter_px"] = branch_diameter_px
        record["bolus_curve"]["bolus_gain_by_frame"] = [round(phase_gain[phase] * (0.64 if realism == "v4" and "low_contrast" in failure_modes else 0.72 if "low_contrast" in failure_modes else 1.0), 3) for phase in phases]
        record["appearance_model"]["detector_artifacts"] = detector_artifacts
    if stress_protocol is not None:
        record["appearance_model"]["stress_protocol"] = stress_protocol
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("research/synthetic_dsa/outputs"))
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--realism", choices=["v0", "v1", "v2", "v3", "v4"], default="v0")
    ap.add_argument("--tag", default=None, help="sequence/manifest tag; defaults to toy for v0, toy_v1 for v1, toy_v2 for v2, toy_v3 for v3, and toy_v4 for v4")
    args = ap.parse_args()
    tag = args.tag or ("toy_v4" if args.realism == "v4" else "toy_v3" if args.realism == "v3" else "toy_v2" if args.realism == "v2" else "toy_v1" if args.realism == "v1" else "toy")
    manifest_dir = args.out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    records = [make_sequence(i + 1, args.seed + i, args.out, args.size, args.frames, realism=args.realism, tag=tag) for i in range(args.count)]
    manifest_path = manifest_dir / f"{tag}_{args.count}_manifest.jsonl"
    with manifest_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "count": len(records), "out": str(args.out), "realism": args.realism, "tag": tag}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
