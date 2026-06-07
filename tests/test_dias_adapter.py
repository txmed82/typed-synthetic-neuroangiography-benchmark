import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_dias.py"
spec = importlib.util.spec_from_file_location("prepare_dias", SCRIPT)
prepare_dias = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_dias)

BASELINE_SCRIPT = ROOT / "scripts" / "run_dias_segmentation_baseline.py"
baseline_spec = importlib.util.spec_from_file_location("run_dias_segmentation_baseline", BASELINE_SCRIPT)
run_dias_segmentation_baseline = importlib.util.module_from_spec(baseline_spec)
baseline_spec.loader.exec_module(run_dias_segmentation_baseline)


def make_img(path: Path, pixels, size=(12, 12), value=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", size, 0)
    for x, y in pixels:
        img.putpixel((x, y), value)
    img.save(path)


class DiasAdapterTests(unittest.TestCase):
    def make_tiny_dias(self, root: Path):
        train_img = root / "training" / "images"
        train_lab = root / "training" / "labels"
        val_img = root / "validation" / "images"
        val_lab = root / "validation" / "labels"
        # sequence 0 has two frames and one sequence-level mask
        make_img(train_img / "image_s0_i0.png", [(2, 2), (2, 3)], value=80)
        make_img(train_img / "image_s0_i1.png", [(2, 2), (2, 3), (3, 3)], value=220)
        make_img(train_lab / "label_s0.png", [(2, 2), (2, 3), (3, 3)], value=255)
        # validation/eval sequence
        make_img(val_img / "image_s30_i0.png", [(5, 5), (5, 6)], value=90)
        make_img(val_img / "image_s30_i1.png", [(5, 5), (5, 6), (6, 6)], value=230)
        make_img(val_lab / "label_s30.png", [(5, 5), (5, 6), (6, 6)], value=255)

    def test_parse_dias_image_name_extracts_sequence_and_frame(self):
        parsed = prepare_dias.parse_dias_image_name("image_s40_i7.png")
        self.assertEqual(parsed, ("40", 7))

    def test_build_manifest_groups_frames_by_sequence_with_label(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_tiny_dias(root)
            records = prepare_dias.build_manifest_records(root, splits=["training", "validation"])

            self.assertEqual(len(records), 2)
            first = records[0]
            self.assertEqual(first["source_dataset"], "DIAS")
            self.assertEqual(first["split"], "training")
            self.assertEqual(first["sequence_id"], "dias_s0")
            self.assertEqual(first["dsa_frame_sequence"]["frame_count"], 2)
            self.assertEqual(first["vessel_mask_sequence"]["uri"], "training/labels/label_s0.png")
            self.assertEqual(first["frame_files"], ["training/images/image_s0_i0.png", "training/images/image_s0_i1.png"])

    def test_threshold_baseline_trains_on_training_and_evaluates_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_tiny_dias(root)
            manifest = root / "dias_manifest.jsonl"
            prepare_dias.write_manifest(prepare_dias.build_manifest_records(root, splits=["training", "validation"]), manifest)

            report = run_dias_segmentation_baseline.run_baseline(manifest, root, train_split="training", eval_split="validation")

            self.assertEqual(report["dataset"], "DIAS")
            self.assertEqual(report["train_sequence_count"], 1)
            self.assertEqual(report["eval_sequence_count"], 1)
            self.assertGreaterEqual(report["model"]["threshold"], 1)
            self.assertTrue(math.isclose(report["aggregate"]["mean_iou"], 1.0))
            self.assertTrue(math.isclose(report["aggregate"]["mean_dice"], 1.0))
            self.assertEqual(report["per_sequence"][0]["sequence_id"], "dias_s30")
    def test_morphology_baseline_exposes_postprocess_parameters(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_tiny_dias(root)
            manifest = root / "dias_manifest.jsonl"
            prepare_dias.write_manifest(prepare_dias.build_manifest_records(root, splits=["training", "validation"]), manifest)

            report = run_dias_segmentation_baseline.run_baseline(
                manifest,
                root,
                train_split="training",
                eval_split="validation",
                baseline="projection_morphology",
            )

            self.assertEqual(report["model"]["name"], "projection_morphology")
            self.assertIn("postprocess_search", report["model"])
            self.assertIn("open_radius", report["per_sequence"][0])
            self.assertTrue(math.isclose(report["aggregate"]["mean_dice"], 1.0))

    def test_remove_small_components_drops_speckles(self):
        img = Image.new("L", (8, 8), 0)
        img.putpixel((0, 0), 255)
        for xy in [(3, 3), (3, 4), (4, 3), (4, 4)]:
            img.putpixel(xy, 255)

        cleaned = run_dias_segmentation_baseline.remove_small_components(img, min_area=2)

        self.assertEqual(cleaned.getpixel((0, 0)), 0)
        self.assertEqual(cleaned.getpixel((3, 3)), 255)


if __name__ == "__main__":
    unittest.main()
