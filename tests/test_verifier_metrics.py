import importlib.util
import math
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verifier_metrics.py"
spec = importlib.util.spec_from_file_location("verifier_metrics", SCRIPT)
verifier_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier_metrics)


def make_mask(path: Path, size=(8, 8), pixels=()):
    image = Image.new("L", size, 0)
    for x, y in pixels:
        image.putpixel((x, y), 255)
    image.save(path)


class VerifierMetricsTests(unittest.TestCase):
    def test_compute_iou_and_dice_for_binary_masks(self):
        with self.subTest("binary mask overlap"):
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                tmp_path = Path(d)
                truth = tmp_path / "truth.png"
                pred = tmp_path / "pred.png"
                make_mask(truth, pixels=[(1, 1), (1, 2), (2, 1), (2, 2)])
                make_mask(pred, pixels=[(1, 1), (1, 2), (3, 3), (4, 4)])

                metrics = verifier_metrics.binary_mask_metrics(truth, pred)

                self.assertEqual(metrics["intersection_px"], 2)
                self.assertEqual(metrics["union_px"], 6)
                self.assertEqual(metrics["truth_area_px"], 4)
                self.assertEqual(metrics["pred_area_px"], 4)
                self.assertTrue(math.isclose(metrics["iou"], 2 / 6))
                self.assertTrue(math.isclose(metrics["dice"], 4 / 8))

    def test_tip_localization_metrics_include_threshold_hits(self):
        metrics = verifier_metrics.tip_localization_metrics(
            truth_xy=[10.0, 10.0],
            pred_xy=[13.0, 14.0],
            thresholds=(2, 5, 10),
        )

        self.assertTrue(math.isclose(metrics["tip_error_px"], 5.0))
        self.assertIs(metrics["within_2px"], False)
        self.assertIs(metrics["within_5px"], True)
        self.assertIs(metrics["within_10px"], True)

    def test_bolus_phase_metrics_report_accuracy_and_mae(self):
        metrics = verifier_metrics.bolus_phase_metrics(
            truth_phases=["precontrast", "arrival", "arterial_peak", "washout"],
            pred_phases=["precontrast", "arterial_peak", "arterial_peak", "arrival"],
        )

        self.assertTrue(math.isclose(metrics["phase_accuracy"], 0.5))
        self.assertTrue(math.isclose(metrics["phase_mae_frames"], 0.75))
        self.assertEqual(metrics["phase_confusion"]["arrival->arterial_peak"], 1)
        self.assertEqual(metrics["phase_confusion"]["washout->arrival"], 1)

    def toy_record(self, tmp_path):
        seq = tmp_path / "outputs" / "sequences" / "sdsa_toy_test"
        frames = seq / "frames"
        masks = seq / "masks"
        frames.mkdir(parents=True)
        masks.mkdir(parents=True)
        frame0 = Image.new("L", (8, 8), 0)
        frame0.putpixel((1, 1), 230)
        frame0.putpixel((2, 2), 230)
        frame0.save(frames / "frame_000.png")
        frame1 = Image.new("L", (8, 8), 0)
        frame1.putpixel((1, 1), 160)
        frame1.putpixel((2, 2), 160)
        frame1.putpixel((3, 3), 160)
        frame1.putpixel((6, 6), 255)
        frame1.save(frames / "frame_001.png")
        make_mask(masks / "mask_000.png", pixels=[(1, 1), (2, 2)])
        make_mask(masks / "mask_001.png", pixels=[(1, 1), (2, 2), (3, 3)])
        return {
            "sequence_id": "sdsa_toy_test",
            "dsa_frame_sequence": {"uri": "outputs/sequences/sdsa_toy_test/frames", "frame_count": 2, "height": 8, "width": 8},
            "vessel_mask_sequence": {"uri": "outputs/sequences/sdsa_toy_test/masks", "frame_count": 2, "height": 8, "width": 8},
            "bolus_curve": {"phase_by_frame": ["precontrast", "arrival"]},
            "catheter_tip_state": {"tip_xy_by_frame": [[1, 1], [2, 2]], "visibility_by_frame": ["visible", "visible"]},
            "catheter_path": {"occlusion_flags_by_frame": [False, False]},
            "failure_modes": ["noise"],
        }

    def test_manifest_verification_summarizes_identity_baseline(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)
            record = self.toy_record(tmp_path)

            summary = verifier_metrics.verify_record(record, tmp_path, baseline="identity")

            self.assertEqual(summary["sequence_id"], "sdsa_toy_test")
            self.assertEqual(summary["frame_count"], 2)
            self.assertTrue(math.isclose(summary["mean_iou"], 1.0))
            self.assertTrue(math.isclose(summary["mean_dice"], 1.0))
            self.assertTrue(math.isclose(summary["mean_tip_error_px"], 0.0))
            self.assertTrue(math.isclose(summary["tip_within_5px_rate"], 1.0))
            self.assertTrue(math.isclose(summary["phase_accuracy"], 1.0))
            self.assertEqual(summary["failure_modes"], ["noise"])

    def test_frame_threshold_baseline_produces_non_oracle_metrics(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)
            record = self.toy_record(tmp_path)

            summary = verifier_metrics.verify_record(record, tmp_path, baseline="frame_threshold")

            self.assertLess(summary["mean_iou"], 1.0)
            self.assertGreater(summary["mean_tip_error_px"], 0.0)
            self.assertIn("baseline_params", summary)


if __name__ == "__main__":
    unittest.main()
