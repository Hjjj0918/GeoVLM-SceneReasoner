"""Test detection label normalization before segmentation and geometry stages.

Usage: python -m pytest tests/test_normalize_detections.py -q
"""

import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_normalize_module():
    module_path = REPO_ROOT / "scripts" / "06_normalize_detections.py"
    spec = importlib.util.spec_from_file_location("normalize_detections", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NormalizeDetectionsTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.detection_dir = self.tmp_path / "outputs" / "detections"
        self.output_dir = self.tmp_path / "outputs" / "detections_normalized"
        self.detection_dir.mkdir(parents=True)
        self.module = load_normalize_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_detection(self, name: str, payload: dict) -> Path:
        path = self.detection_dir / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def detection_payload(self) -> dict:
        return {
            "image": "scene_0001_view_15.jpg",
            "image_width": 5712,
            "image_height": 4284,
            "model": "yolo11n.pt",
            "objects": [
                {
                    "object_id": "obj_001",
                    "label": "book",
                    "confidence": 0.309927,
                    "bbox_xyxy": [600.2, 1195.8, 4357.2, 2852.8],
                },
                {
                    "object_id": "obj_002",
                    "label": "mouse",
                    "confidence": 0.730836,
                    "bbox_xyxy": [5107.9, 1931.7, 5710.3, 2766.4],
                },
            ],
        }

    def correction_payload(self) -> dict:
        return {
            "version": "0.1",
            "description": "Test corrections.",
            "global_label_aliases": {"book": "laptop"},
        }

    def test_collect_detection_files_sorts_json_files(self):
        self.write_detection("scene_0001_view_01.json", self.detection_payload())
        self.write_detection("notes.txt", {})
        self.write_detection("scene_0001_view_00.json", self.detection_payload())

        files = self.module.collect_detection_files(self.detection_dir)

        self.assertEqual(
            [path.name for path in files],
            ["scene_0001_view_00.json", "scene_0001_view_01.json"],
        )

    def test_normalize_detection_payload_applies_alias_and_preserves_raw_label(self):
        normalized = self.module.normalize_detection_payload(
            self.detection_payload(),
            correction_payload=self.correction_payload(),
        )

        laptop = normalized["objects"][0]
        mouse = normalized["objects"][1]
        self.assertEqual(laptop["label"], "laptop")
        self.assertEqual(laptop["raw_label"], "book")
        self.assertEqual(laptop["normalization"], {"rule": "global_label_aliases", "from": "book", "to": "laptop"})
        self.assertEqual(mouse["label"], "mouse")
        self.assertNotIn("raw_label", mouse)

    def test_image_specific_alias_overrides_global_alias(self):
        correction_payload = {
            "global_label_aliases": {"book": "laptop"},
            "image_label_aliases": {
                "scene_0001_view_15.jpg": {"book": "notebook"}
            },
        }

        normalized = self.module.normalize_detection_payload(
            self.detection_payload(),
            correction_payload=correction_payload,
        )

        self.assertEqual(normalized["objects"][0]["label"], "notebook")
        self.assertEqual(normalized["objects"][0]["normalization"]["rule"], "image_label_aliases")

    def test_write_normalized_payload_rejects_existing_output_without_overwrite(self):
        output_path = self.output_dir / "scene_0001_view_15.json"
        output_path.parent.mkdir(parents=True)
        output_path.write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.module.write_normalized_payload({"objects": []}, output_path, overwrite=False)

    def test_normalize_detection_file_writes_output(self):
        detection_path = self.write_detection("scene_0001_view_15.json", self.detection_payload())

        output_path = self.module.normalize_detection_file(
            detection_path=detection_path,
            output_dir=self.output_dir,
            correction_payload=self.correction_payload(),
            overwrite=False,
        )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["objects"][0]["label"], "laptop")
        self.assertEqual(payload["objects"][0]["raw_label"], "book")


if __name__ == "__main__":
    unittest.main()
