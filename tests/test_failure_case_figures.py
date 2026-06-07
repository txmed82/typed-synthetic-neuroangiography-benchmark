import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_failure_case_figures.py"
spec = importlib.util.spec_from_file_location("make_failure_case_figures", SCRIPT)
figs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figs)


class FailureCaseFigureTests(unittest.TestCase):
    def test_overlay_frame_marks_mask_and_tip_in_rgb(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame = root / "frame_000.png"
            mask = root / "mask_000.png"
            Image.new("L", (12, 12), 20).save(frame)
            m = Image.new("L", (12, 12), 0)
            m.putpixel((4, 10), 255)
            m.save(mask)

            out = figs.overlay_frame(frame, mask, [8, 8])

            self.assertEqual(out.mode, "RGB")
            self.assertGreater(out.getpixel((4, 10))[1], out.getpixel((4, 10))[0])
            self.assertGreater(out.getpixel((8, 8))[0], 200)

    def test_overlay_frame_marks_predicted_tip_in_cyan_separate_from_truth(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frame = root / "frame_000.png"
            mask = root / "mask_000.png"
            Image.new("L", (16, 16), 20).save(frame)
            Image.new("L", (16, 16), 0).save(mask)

            out = figs.overlay_frame(frame, mask, [4, 4], predicted_tip_xy=[11, 11])

            self.assertGreater(out.getpixel((4, 4))[0], 200)
            self.assertGreater(out.getpixel((11, 11))[1], 180)
            self.assertGreater(out.getpixel((11, 11))[2], 180)
            self.assertLess(out.getpixel((11, 11))[0], 100)

    def test_make_panel_uses_predicted_tip_from_worst_frame_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            seq = root / "outputs" / "sequences" / "s1"
            frames = seq / "frames"
            masks = seq / "masks"
            frames.mkdir(parents=True)
            masks.mkdir(parents=True)
            Image.new("L", (16, 16), 20).save(frames / "frame_000.png")
            Image.new("L", (16, 16), 0).save(masks / "mask_000.png")
            record = {
                "sequence_id": "s1",
                "dsa_frame_sequence": {"uri": "outputs/sequences/s1/frames", "frame_count": 1},
                "vessel_mask_sequence": {"uri": "outputs/sequences/s1/masks", "frame_count": 1},
                "catheter_tip_state": {"tip_xy_by_frame": [[4, 4]]},
            }
            summary = {
                "sequence_id": "s1",
                "mean_iou": 0.5,
                "mean_tip_error_px": 7.0,
                "phase_accuracy": 1.0,
                "per_frame": [{"frame_index": 0, "iou": 0.5, "tip_error_px": 7.0, "predicted_tip_xy": [11, 11]}],
            }

            panel = figs.make_panel(record, summary, root=root, thumb_size=(16, 16))

            self.assertGreater(panel.getpixel((11, 11))[1], 180)
            self.assertGreater(panel.getpixel((11, 11))[2], 180)

    def test_select_worst_sequences_prefers_low_iou_then_high_tip_error(self):
        report = {
            "sequences": [
                {"sequence_id": "good", "mean_iou": 0.9, "mean_tip_error_px": 0.5},
                {"sequence_id": "bad_iou", "mean_iou": 0.2, "mean_tip_error_px": 0.2},
                {"sequence_id": "bad_tip", "mean_iou": 0.8, "mean_tip_error_px": 9.0},
            ]
        }

        selected = figs.select_worst_sequences(report, metric="composite", limit=2)

        self.assertEqual([s["sequence_id"] for s in selected], ["bad_iou", "bad_tip"])


if __name__ == "__main__":
    unittest.main()
