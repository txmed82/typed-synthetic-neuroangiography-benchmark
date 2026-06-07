import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_dias_overlay_figures.py"
spec = importlib.util.spec_from_file_location("make_dias_overlay_figures", SCRIPT)
make_dias_overlay_figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_dias_overlay_figures)


def make_img(path: Path, pixels, size=(16, 16), value=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", size, 0)
    for x, y in pixels:
        img.putpixel((x, y), value)
    img.save(path)


class DiasOverlayFigureTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        dataset = root / "dias"
        frame_dir = dataset / "test" / "images"
        label_dir = dataset / "test" / "labels"
        pred_dir = root / "pred"
        truth = [(4, 4), (4, 5), (5, 5), (8, 8)]
        make_img(frame_dir / "image_s40_i0.png", [(4, 4), (4, 5)], value=80)
        make_img(frame_dir / "image_s40_i1.png", truth, value=220)
        make_img(label_dir / "label_s40.png", truth, value=255)
        make_img(pred_dir / "threshold.png", [(4, 4), (4, 5), (7, 7)], value=255)
        make_img(pred_dir / "morph.png", truth, value=255)
        manifest = root / "dias_manifest.jsonl"
        record = {
            "sequence_id": "dias_s40",
            "split": "test",
            "has_labels": True,
            "dsa_frame_sequence": {"frame_count": 2, "uri": "unused"},
            "vessel_mask_sequence": {"uri": "test/labels/label_s40.png"},
            "frame_files": ["test/images/image_s40_i0.png", "test/images/image_s40_i1.png"],
        }
        manifest.write_text(json.dumps(record) + "\n")
        threshold_report = root / "threshold.json"
        morphology_report = root / "morphology.json"
        threshold_report.write_text(json.dumps({
            "eval_split": "test",
            "aggregate": {"mean_dice": 0.5},
            "per_sequence": [{
                "sequence_id": "dias_s40", "iou": 0.4, "dice": 0.5,
                "truth_area_px": 4, "pred_area_px": 3,
                "prediction_uri": str(pred_dir / "threshold.png"), "projection": "range",
            }],
        }))
        morphology_report.write_text(json.dumps({
            "eval_split": "test",
            "aggregate": {"mean_dice": 1.0},
            "per_sequence": [{
                "sequence_id": "dias_s40", "iou": 1.0, "dice": 1.0,
                "truth_area_px": 4, "pred_area_px": 4,
                "prediction_uri": str(pred_dir / "morph.png"), "projection": "range",
            }],
        }))
        return manifest, dataset, threshold_report, morphology_report

    def test_select_sequences_prefers_hard_and_delta_cases(self):
        threshold_report = {"per_sequence": [{"sequence_id": "a", "dice": 0.5}, {"sequence_id": "b", "dice": 0.9}]}
        morphology_report = {"per_sequence": [{"sequence_id": "a", "dice": 0.6}, {"sequence_id": "b", "dice": 0.2}]}

        selected = make_dias_overlay_figures.select_sequences(threshold_report, morphology_report, limit=2)

        self.assertEqual(selected[0], "b")
        self.assertEqual(set(selected), {"a", "b"})

    def test_make_comparison_sheet_writes_png_and_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, dataset, threshold_report, morphology_report = self.make_fixture(root)
            out_png = root / "figure.png"
            out_json = root / "figure.json"

            metadata = make_dias_overlay_figures.make_comparison_sheet(
                manifest,
                dataset,
                threshold_report,
                morphology_report,
                out_png,
                out_json,
                limit=1,
            )

            self.assertTrue(out_png.exists())
            self.assertTrue(out_json.exists())
            self.assertEqual(metadata["selected"][0]["sequence_id"], "dias_s40")
            self.assertAlmostEqual(metadata["selected"][0]["dice_delta"], 0.5)
            with Image.open(out_png) as img:
                self.assertGreater(img.size[0], 0)
                self.assertGreater(img.size[1], 0)


if __name__ == "__main__":
    unittest.main()
