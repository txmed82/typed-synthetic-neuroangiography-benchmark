import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_toy_sequences.py"
spec = importlib.util.spec_from_file_location("generate_toy_sequences", SCRIPT)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class GeneratorRealismTests(unittest.TestCase):
    def test_v1_sequence_records_realism_metadata_and_more_vessels(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "outputs"
            record = generator.make_sequence(1, 20260606, out, size=96, frames=6, realism="v1", tag="v1test")

            self.assertEqual(record["sequence_id"], "sdsa_v1test_0001")
            self.assertEqual(record["generator"]["version"], "0.2.0")
            self.assertIn("appearance_model", record)
            self.assertGreaterEqual(record["vascular_graph"]["edge_count"], 7)
            self.assertGreaterEqual(len(record["vascular_graph"]["branch_labels"]), 5)
            self.assertEqual(len(record["appearance_model"]["motion_shift_px_by_frame"]), 6)
            self.assertTrue(any(abs(x) > 0 or abs(y) > 0 for x, y in record["appearance_model"]["motion_shift_px_by_frame"]))
            self.assertTrue((out / "sequences" / "sdsa_v1test_0001" / "frames" / "frame_000.png").exists())
            self.assertTrue((out / "sequences" / "sdsa_v1test_0001" / "masks" / "mask_000.png").exists())

    def test_v2_sequence_records_device_masks_and_dense_catheter_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "outputs"
            record = generator.make_sequence(1, 20260606, out, size=96, frames=6, realism="v2", tag="v2test")

            self.assertEqual(record["generator"]["version"], "0.3.0")
            self.assertEqual(record["appearance_model"]["realism"], "v2")
            self.assertIn("device_mask_sequence", record)
            self.assertGreater(len(record["catheter_path"]["polyline_px"]), 25)
            self.assertTrue((out / "sequences" / "sdsa_v2test_0001" / "device_masks" / "device_000.png").exists())

    def test_v3_sequence_records_richer_realism_metadata_and_artifacts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "outputs"
            record = generator.make_sequence(1, 20260606, out, size=96, frames=6, realism="v3", tag="v3test")

            self.assertEqual(record["generator"]["version"], "0.4.0")
            self.assertEqual(record["appearance_model"]["realism"], "v3")
            self.assertIn("device_mask_sequence", record)
            self.assertIn("branch_diameter_px", record["vascular_graph"])
            self.assertGreater(len(record["vascular_graph"]["branch_diameter_px"]), 5)
            self.assertIn("bolus_gain_by_frame", record["bolus_curve"])
            self.assertEqual(len(record["bolus_curve"]["bolus_gain_by_frame"]), 6)
            self.assertIn("detector_artifacts", record["appearance_model"])
            self.assertIn("bone_shadow_count", record["appearance_model"]["detector_artifacts"])
            self.assertIn("coil_mass", record["failure_modes"])
            self.assertTrue((out / "sequences" / "sdsa_v3test_0001" / "device_masks" / "device_000.png").exists())

    def test_v4_sequence_records_coil_projection_stress_protocol(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "outputs"
            record = generator.make_sequence(1, 20260606, out, size=96, frames=6, realism="v4", tag="v4stress")

            self.assertEqual(record["generator"]["version"], "0.5.0")
            self.assertEqual(record["appearance_model"]["realism"], "v4")
            self.assertIn("device_mask_sequence", record)
            self.assertIn("stress_protocol", record["appearance_model"])
            protocol = record["appearance_model"]["stress_protocol"]
            self.assertEqual(protocol["name"], "coil_projection_ambiguity_v0")
            self.assertGreaterEqual(protocol["coil_decoy_count"], 3)
            self.assertGreaterEqual(protocol["projection_ambiguity_score"], 0.75)
            self.assertLess(protocol["catheter_salience"], 0.75)
            self.assertIn("coil_mass", record["failure_modes"])
            self.assertIn("projection_ambiguity", record["failure_modes"])
            self.assertIn("bolus_gain_by_frame", record["bolus_curve"])
            self.assertTrue((out / "sequences" / "sdsa_v4stress_0001" / "device_masks" / "device_000.png").exists())


if __name__ == "__main__":
    unittest.main()
