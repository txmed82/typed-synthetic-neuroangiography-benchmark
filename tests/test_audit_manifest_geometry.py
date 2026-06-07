import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_manifest_geometry.py"
spec = importlib.util.spec_from_file_location("audit_manifest_geometry", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write_img(path: Path, size=(16, 16), points=(), value=255):
    img = Image.new("L", size, 0)
    for x, y in points:
        img.putpixel((x, y), value)
    img.save(path)


class AuditManifestGeometryTests(unittest.TestCase):
    def make_record(self, root: Path, sequence_id="qa_ok", tip=(8, 8), device_points=None):
        seq = root / "outputs" / "sequences" / sequence_id
        frames = seq / "frames"
        masks = seq / "masks"
        devices = seq / "device_masks"
        frames.mkdir(parents=True)
        masks.mkdir(parents=True)
        devices.mkdir(parents=True)
        vessel = [(4, 4), (5, 5), (6, 6)]
        device_points = device_points if device_points is not None else [(7, 8), (8, 8), (9, 8)]
        write_img(frames / "frame_000.png", points=vessel + device_points, value=240)
        write_img(masks / "mask_000.png", points=vessel, value=255)
        write_img(devices / "device_000.png", points=device_points, value=255)
        return {
            "sequence_id": sequence_id,
            "dsa_frame_sequence": {"uri": f"outputs/sequences/{sequence_id}/frames", "frame_count": 1, "height": 16, "width": 16, "dtype": "uint8"},
            "vessel_mask_sequence": {"uri": f"outputs/sequences/{sequence_id}/masks", "frame_count": 1, "height": 16, "width": 16, "dtype": "bool"},
            "device_mask_sequence": {"uri": f"outputs/sequences/{sequence_id}/device_masks", "frame_count": 1, "height": 16, "width": 16, "dtype": "bool"},
            "catheter_path": {"path_type": "centerline_following", "polyline_px": [[7, 8], [8, 8], [9, 8]], "occlusion_flags_by_frame": [False]},
            "catheter_tip_state": {"tip_xy_by_frame": [[tip[0], tip[1]]], "visibility_by_frame": ["visible"], "confidence_target_by_frame": [0.95]},
            "bolus_curve": {"phase_by_frame": ["arterial_peak"]},
            "failure_modes": [],
        }

    def test_audit_record_passes_when_tip_lands_on_device_mask(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record = self.make_record(root)
            result = audit.audit_record(record, root)
            self.assertTrue(result["passes_qa"])
            self.assertEqual(result["frames"][0]["tip_to_device_px"], 0.0)
            self.assertGreater(result["device_presence_rate"], 0.99)

    def test_audit_record_fails_when_visible_tip_is_far_from_device_mask(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record = self.make_record(root, tip=(1, 1))
            result = audit.audit_record(record, root)
            self.assertFalse(result["passes_qa"])
            self.assertGreater(result["frames"][0]["tip_to_device_px"], 5.0)
            self.assertIn("tip_not_on_device", result["qa_failures"])

    def test_diagnostic_overlay_has_distinct_vessel_device_tip_colors(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            record = self.make_record(root)
            out = audit.render_diagnostic_overlay(record, root, frame_index=0)
            self.assertEqual(out.mode, "RGB")
            self.assertGreater(out.getpixel((4, 4))[1], out.getpixel((4, 4))[0])
            self.assertGreater(out.getpixel((8, 8))[0], 200)


if __name__ == "__main__":
    unittest.main()
