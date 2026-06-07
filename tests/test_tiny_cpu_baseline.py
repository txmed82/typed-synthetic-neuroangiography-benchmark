import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_tiny_cpu_baseline.py"
spec = importlib.util.spec_from_file_location("run_tiny_cpu_baseline", SCRIPT)
tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny)


def write_frame(path: Path, size=(8, 8), bright_pixels=(), noise_pixels=()):
    image = Image.new("L", size, 0)
    for x, y, val in noise_pixels:
        image.putpixel((x, y), val)
    for x, y, val in bright_pixels:
        image.putpixel((x, y), val)
    image.save(path)


def write_mask(path: Path, size=(8, 8), pixels=()):
    image = Image.new("L", size, 0)
    for x, y in pixels:
        image.putpixel((x, y), 255)
    image.save(path)


class TinyCpuBaselineTests(unittest.TestCase):
    def make_record(self, root: Path, sequence_id: str, vessel_val: int, noise_val: int, device_pixels=None, tip_xy=None):
        seq = root / "outputs" / "sequences" / sequence_id
        frames = seq / "frames"
        masks = seq / "masks"
        device_masks = seq / "device_masks"
        frames.mkdir(parents=True)
        masks.mkdir(parents=True)
        if device_pixels is not None:
            device_masks.mkdir(parents=True)
        vessel_pixels = [(1, 1), (2, 2), (3, 3)]
        tip_xy = tip_xy or [3, 3]
        bright_pixels = [(x, y, vessel_val) for x, y in vessel_pixels]
        if device_pixels is not None:
            bright_pixels.extend((x, y, 210) for x, y in device_pixels)
        write_frame(frames / "frame_000.png", bright_pixels=bright_pixels, noise_pixels=[(6, 6, noise_val)])
        write_mask(masks / "mask_000.png", pixels=vessel_pixels)
        record = {
            "sequence_id": sequence_id,
            "dsa_frame_sequence": {"uri": f"outputs/sequences/{sequence_id}/frames", "frame_count": 1, "height": 8, "width": 8},
            "vessel_mask_sequence": {"uri": f"outputs/sequences/{sequence_id}/masks", "frame_count": 1, "height": 8, "width": 8},
            "bolus_curve": {"phase_by_frame": ["arterial_peak"]},
            "catheter_tip_state": {"tip_xy_by_frame": [tip_xy], "visibility_by_frame": ["visible"]},
            "catheter_path": {"occlusion_flags_by_frame": [False]},
            "failure_modes": [],
        }
        if device_pixels is not None:
            write_mask(device_masks / "device_000.png", pixels=device_pixels)
            record["device_mask_sequence"] = {"uri": f"outputs/sequences/{sequence_id}/device_masks", "frame_count": 1, "height": 8, "width": 8}
        return record

    def test_learns_threshold_from_training_masks_and_evaluates_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            train = [self.make_record(root, "sdsa_train", vessel_val=170, noise_val=120)]
            eval_records = [self.make_record(root, "sdsa_eval", vessel_val=172, noise_val=130)]

            model = tiny.train_intensity_model(train, root)
            summary = tiny.evaluate_records(eval_records, root, model)

            self.assertLess(model["threshold"], 170)
            self.assertGreater(model["threshold"], 130)
            self.assertTrue(math.isclose(summary["aggregate"]["mean_iou"], 1.0))
            self.assertEqual(summary["sequence_count"], 1)
            self.assertEqual(summary["model"]["name"], "tiny_intensity_cpu_baseline")
            self.assertIn("predicted_tip_xy", summary["sequences"][0]["per_frame"][0])
    def test_temporal_phase_rule_uses_peak_frame_and_order(self):
        phases = tiny.temporal_phase_predictions([10.0, 20.0, 35.0, 90.0, 55.0, 30.0])
        self.assertEqual(phases, ["precontrast", "arrival", "arrival", "arterial_peak", "washout", "washout"])

    def test_build_train_eval_sets_supports_cross_manifest_training(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            train_records = [self.make_record(root, "sdsa_train_a", vessel_val=170, noise_val=120)]
            eval_records = [self.make_record(root, "sdsa_eval_b", vessel_val=172, noise_val=130)]
            train_manifest = root / "train.jsonl"
            eval_manifest = root / "eval.jsonl"
            train_manifest.write_text("\n".join(json.dumps(r) for r in train_records) + "\n")
            eval_manifest.write_text("\n".join(json.dumps(r) for r in eval_records) + "\n")

            train, eval_rows, split = tiny.build_train_eval_sets(
                eval_manifest=eval_manifest,
                train_manifests=[train_manifest],
                train_fraction=0.7,
            )

            self.assertEqual([r["sequence_id"] for r in train], ["sdsa_train_a"])
            self.assertEqual([r["sequence_id"] for r in eval_rows], ["sdsa_eval_b"])
            self.assertEqual(split["mode"], "cross_manifest")
            self.assertEqual(split["train_manifests"], [str(train_manifest)])
            self.assertEqual(split["eval_manifest"], str(eval_manifest))

    def test_trains_heatmap_model_from_device_masks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            train = [self.make_record(root, "sdsa_train_device", vessel_val=160, noise_val=80, device_pixels=[(4, 4), (5, 5)], tip_xy=[5, 5])]

            model = tiny.train_heatmap_model(train, root)

            self.assertEqual(model["name"], "tiny_heatmap_cpu_baseline")
            self.assertGreaterEqual(model["device_threshold"], 161)
            self.assertEqual(model["train_device_mask_sequence_count"], 1)
            self.assertIn("tip_spatial_prior_by_frame_norm", model)

    def test_heatmap_tip_prediction_prefers_learned_device_endpoint_over_bright_distractor(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record = self.make_record(
                root,
                "sdsa_eval_device",
                vessel_val=180,
                noise_val=250,
                device_pixels=[(4, 4), (5, 5), (6, 6)],
                tip_xy=[6, 6],
            )
            model = {
                "name": "tiny_heatmap_cpu_baseline",
                "threshold": 150,
                "device_threshold": 200,
                "tip_spatial_prior_by_frame_norm": [[6 / 7, 6 / 7]],
                "tip_prior_weight": 4.0,
                "tip_intensity_weight": 0.01,
                "tip_endpoint_weight": 2.0,
                "phase_rule": "temporal_rank",
            }

            pred = tiny.predict_record(record, root, model)

            self.assertLess(math.hypot(pred["tip_xy_by_frame"][0][0] - 6, pred["tip_xy_by_frame"][0][1] - 6), 1.1)

    def make_patch_tip_record(self, root: Path, sequence_id: str, tip_xy=(7, 7), distractor_xy=None):
        seq = root / "outputs" / "sequences" / sequence_id
        frames = seq / "frames"
        masks = seq / "masks"
        devices = seq / "device_masks"
        frames.mkdir(parents=True)
        masks.mkdir(parents=True)
        devices.mkdir(parents=True)
        frame = Image.new("L", (15, 15), 0)
        device = Image.new("L", (15, 15), 0)
        tx, ty = tip_xy
        for x, y, value in [(tx, ty, 210), (tx - 1, ty, 185), (tx + 1, ty, 185), (tx, ty - 1, 185), (tx, ty + 1, 185)]:
            frame.putpixel((x, y), value)
            device.putpixel((x, y), 255)
        if distractor_xy is not None:
            dx, dy = distractor_xy
            frame.putpixel((dx, dy), 245)
            device.putpixel((dx, dy), 255)
        frame.save(frames / "frame_000.png")
        write_mask(masks / "mask_000.png", size=(15, 15), pixels=[(tx, ty)])
        device.save(devices / "device_000.png")
        return {
            "sequence_id": sequence_id,
            "dsa_frame_sequence": {"uri": f"outputs/sequences/{sequence_id}/frames", "frame_count": 1, "height": 15, "width": 15},
            "vessel_mask_sequence": {"uri": f"outputs/sequences/{sequence_id}/masks", "frame_count": 1, "height": 15, "width": 15},
            "device_mask_sequence": {"uri": f"outputs/sequences/{sequence_id}/device_masks", "frame_count": 1, "height": 15, "width": 15},
            "bolus_curve": {"phase_by_frame": ["arterial_peak"]},
            "catheter_tip_state": {"tip_xy_by_frame": [[tx, ty]], "visibility_by_frame": ["visible"]},
            "catheter_path": {"occlusion_flags_by_frame": [False]},
            "failure_modes": [],
        }

    def test_trains_patch_heatmap_model_with_local_tip_template(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            train = [self.make_patch_tip_record(root, "sdsa_train_patch")]

            model = tiny.train_patch_heatmap_model(train, root)

            self.assertEqual(model["name"], "tiny_patch_heatmap_cpu_baseline")
            self.assertEqual(model["patch_radius"], 2)
            self.assertEqual(len(model["tip_patch_template"]), 25)
            self.assertGreater(model["tip_patch_template"][12], 0.8)
            self.assertEqual(model["train_patch_count"], 1)

    def test_patch_heatmap_prediction_prefers_tip_like_patch_over_brighter_distractor(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            train = [self.make_patch_tip_record(root, "sdsa_train_patch", tip_xy=(7, 7))]
            eval_record = self.make_patch_tip_record(root, "sdsa_eval_patch", tip_xy=(7, 7), distractor_xy=(2, 2))
            model = tiny.train_patch_heatmap_model(train, root)
            model["phase_rule"] = "temporal_rank"

            pred = tiny.predict_record(eval_record, root, model)

            self.assertLess(math.hypot(pred["tip_xy_by_frame"][0][0] - 7, pred["tip_xy_by_frame"][0][1] - 7), 1.1)

    def test_patch_heatmap_uses_device_mask_candidates_below_brightness_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record = self.make_patch_tip_record(root, "sdsa_eval_mask_candidates", tip_xy=(7, 7), distractor_xy=(2, 2))
            frame_path = root / "outputs" / "sequences" / "sdsa_eval_mask_candidates" / "frames" / "frame_000.png"
            image = Image.open(frame_path).convert("L")
            device_path = root / "outputs" / "sequences" / "sdsa_eval_mask_candidates" / "device_masks" / "device_000.png"
            device = Image.new("L", (15, 15), 0)
            # True device pixels are below the model's device threshold; the bright distractor is outside the device mask.
            for xy in [(5, 7), (6, 7), (7, 7)]:
                image.putpixel(xy, 205)
                device.putpixel(xy, 255)
            for xy in [(8, 7), (7, 6), (7, 8)]:
                image.putpixel(xy, 0)
            image.putpixel((2, 2), 255)
            image.save(frame_path)
            device.save(device_path)
            model = {
                "name": "tiny_patch_heatmap_temporal_cpu_baseline",
                "version": "0.4.0",
                "threshold": 240,
                "device_threshold": 240,
                "tip_spatial_prior_by_frame_norm": [[0.5, 0.5]],
                "tip_prior_weight": 4.0,
                "tip_intensity_weight": 0.0,
                "tip_endpoint_weight": 4.0,
                "tip_patch_weight": 0.0,
                "patch_radius": 2,
                "tip_patch_template": [0.0] * 25,
                "phase_centroids": {phase: 0.0 for phase in tiny.PHASES},
                "phase_rule": "temporal_rank",
            }

            pred = tiny.predict_record(record, root, model)

            self.assertLess(math.hypot(pred["tip_xy_by_frame"][0][0] - 7, pred["tip_xy_by_frame"][0][1] - 7), 1.1)

    def make_two_frame_jump_record(self, root: Path, sequence_id: str):
        seq = root / "outputs" / "sequences" / sequence_id
        frames = seq / "frames"
        masks = seq / "masks"
        devices = seq / "device_masks"
        frames.mkdir(parents=True)
        masks.mkdir(parents=True)
        devices.mkdir(parents=True)
        for idx, tip in enumerate([(7, 7), (8, 7)]):
            frame = Image.new("L", (15, 15), 0)
            device = Image.new("L", (15, 15), 0)
            tx, ty = tip
            for x, y, value in [(tx, ty, 210), (tx - 1, ty, 185), (tx + 1, ty, 185), (tx, ty - 1, 185), (tx, ty + 1, 185)]:
                frame.putpixel((x, y), value)
                device.putpixel((x, y), 255)
            if idx == 1:
                # Far decoy is brighter and tip-like, but implausible given frame 0.
                for x, y, value in [(2, 2, 245), (1, 2, 230), (3, 2, 230), (2, 1, 230), (2, 3, 230)]:
                    frame.putpixel((x, y), value)
                    device.putpixel((x, y), 255)
            frame.save(frames / f"frame_{idx:03d}.png")
            device.save(devices / f"device_{idx:03d}.png")
            write_mask(masks / f"mask_{idx:03d}.png", size=(15, 15), pixels=[tip])
        return {
            "sequence_id": sequence_id,
            "dsa_frame_sequence": {"uri": f"outputs/sequences/{sequence_id}/frames", "frame_count": 2, "height": 15, "width": 15},
            "vessel_mask_sequence": {"uri": f"outputs/sequences/{sequence_id}/masks", "frame_count": 2, "height": 15, "width": 15},
            "device_mask_sequence": {"uri": f"outputs/sequences/{sequence_id}/device_masks", "frame_count": 2, "height": 15, "width": 15},
            "bolus_curve": {"phase_by_frame": ["arrival", "arterial_peak"]},
            "catheter_tip_state": {"tip_xy_by_frame": [[7, 7], [8, 7]], "visibility_by_frame": ["visible", "visible"]},
            "catheter_path": {"occlusion_flags_by_frame": [False, False]},
            "failure_modes": [],
        }

    def test_patch_heatmap_sequence_continuity_penalizes_large_frame_to_frame_jumps(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record = self.make_two_frame_jump_record(root, "sdsa_eval_jump")
            model = {
                "name": "tiny_patch_heatmap_temporal_cpu_baseline",
                "version": "0.4.0",
                "threshold": 180,
                "device_threshold": 180,
                "tip_spatial_prior_by_frame_norm": [[0.5, 0.5], [0.5, 0.5]],
                "tip_prior_weight": 0.0,
                "tip_intensity_weight": 0.01,
                "tip_endpoint_weight": 0.0,
                "tip_patch_weight": 0.0,
                "tip_continuity_weight": 20.0,
                "patch_radius": 2,
                "tip_patch_template": [0.0] * 25,
                "phase_centroids": {phase: 0.0 for phase in tiny.PHASES},
                "phase_rule": "temporal_rank",
            }

            pred = tiny.predict_record(record, root, model)

            self.assertLess(math.hypot(pred["tip_xy_by_frame"][1][0] - 8, pred["tip_xy_by_frame"][1][1] - 7), 1.1)

    def test_trains_patch_ranker_model_with_positive_and_negative_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            train = [
                self.make_patch_tip_record(root, "sdsa_train_rank_a", tip_xy=(7, 7), distractor_xy=(2, 2)),
                self.make_patch_tip_record(root, "sdsa_train_rank_b", tip_xy=(8, 7), distractor_xy=(3, 3)),
            ]

            model = tiny.train_patch_ranker_model(train, root, epochs=3)

            self.assertEqual(model["name"], "tiny_patch_ranker_cpu_baseline")
            self.assertGreater(model["train_ranker_positive_count"], 0)
            self.assertGreater(model["train_ranker_negative_count"], 0)
            self.assertEqual(len(model["ranker_weights"]), len(model["ranker_feature_names"]))

    def test_patch_ranker_prediction_prefers_learned_tip_candidate_over_hard_negative(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            train = [
                self.make_patch_tip_record(root, "sdsa_train_rank_a", tip_xy=(7, 7), distractor_xy=(2, 2)),
                self.make_patch_tip_record(root, "sdsa_train_rank_b", tip_xy=(8, 7), distractor_xy=(3, 3)),
            ]
            eval_record = self.make_patch_tip_record(root, "sdsa_eval_rank", tip_xy=(7, 7), distractor_xy=(2, 2))
            model = tiny.train_patch_ranker_model(train, root, epochs=8)
            model["name"] = "tiny_patch_ranker_temporal_cpu_baseline"
            model["phase_rule"] = "temporal_rank"

            pred = tiny.predict_record(eval_record, root, model)

            self.assertLess(math.hypot(pred["tip_xy_by_frame"][0][0] - 7, pred["tip_xy_by_frame"][0][1] - 7), 1.1)

    def test_dynamic_programming_smoother_prefers_plausible_path_over_single_frame_decoy(self):
        candidate_scores = [
            [(7.0, 7.0, 3.0), (2.0, 2.0, 0.0)],
            [(8.0, 7.0, 2.0), (2.0, 2.0, 4.0)],
            [(9.0, 7.0, 3.0), (2.0, 2.0, 0.0)],
        ]

        path = tiny.smooth_candidate_path_dp(candidate_scores, transition_weight=1.0)

        self.assertEqual(path, [[7.0, 7.0], [8.0, 7.0], [9.0, 7.0]])

    def test_patch_ranker_dp_prediction_uses_sequence_smoothing_for_transient_decoy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record = self.make_two_frame_jump_record(root, "sdsa_eval_dp")
            model = {
                "name": "tiny_patch_ranker_dp_temporal_cpu_baseline",
                "version": "0.2.0",
                "threshold": 180,
                "device_threshold": 180,
                "tip_spatial_prior_by_frame_norm": [[0.5, 0.5], [0.5, 0.5]],
                "ranker_weights": [0.0] * 30,
                "ranker_feature_names": ["bias"] + [f"patch_{idx}" for idx in range(25)] + ["intensity_norm", "endpointness", "prior_closeness", "continuity_closeness"],
                "ranker_transition_weight": 20.0,
                "patch_radius": 2,
                "tip_patch_template": [0.0] * 25,
                "phase_centroids": {phase: 0.0 for phase in tiny.PHASES},
                "phase_rule": "temporal_rank",
            }
            # Score mainly by intensity. Frame 1 has a far brighter decoy, so framewise argmax would jump to (2, 2).
            model["ranker_weights"][26] = 10.0

            pred = tiny.predict_record(record, root, model)

            self.assertLess(math.hypot(pred["tip_xy_by_frame"][1][0] - 8, pred["tip_xy_by_frame"][1][1] - 7), 1.1)


if __name__ == "__main__":
    unittest.main()
