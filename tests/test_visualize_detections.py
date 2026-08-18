"""Test detection visualization image generation for manual review.

Usage: python -m pytest tests/test_visualize_detections.py -q
"""

import importlib.util
import json
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_visualize_module():
    module_path = REPO_ROOT / "scripts" / "05_visualize_detections.py"
    spec = importlib.util.spec_from_file_location("visualize_detections", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisualizeDetectionsTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.image_dir = self.tmp_path / "data" / "images"
        self.detection_dir = self.tmp_path / "outputs" / "detections"
        self.output_dir = self.tmp_path / "outputs" / "visualizations" / "detections"
        self.image_dir.mkdir(parents=True)
        self.detection_dir.mkdir(parents=True)
        self.module = load_visualize_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_image(self, name: str, width: int = 120, height: int = 80) -> Path:
        path = self.image_dir / name
        image = np.zeros((height, width, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def write_detection(self, name: str, payload: dict) -> Path:
        path = self.detection_dir / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def detection_payload(self) -> dict:
        return {
            "image": "scene_0001_view_00.jpg",
            "image_width": 120,
            "image_height": 80,
            "model": "yolo11n.pt",
            "objects": [
                {
                    "object_id": "obj_001",
                    "label": "cup",
                    "confidence": 0.81,
                    "bbox_xyxy": [10, 20, 50, 60],
                }
            ],
        }

    def test_label_color_is_stable_for_same_label(self):
        first = self.module.label_color("cup")
        second = self.module.label_color("cup")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_visualize_detection_file_writes_overlay_image(self):
        self.write_image("scene_0001_view_00.jpg")
        detection_path = self.write_detection("scene_0001_view_00.json", self.detection_payload())

        output_path = self.module.visualize_detection_file(
            detection_path=detection_path,
            image_dir=self.image_dir,
            output_dir=self.output_dir,
            overwrite=False,
        )

        self.assertEqual(output_path.name, "scene_0001_view_00.jpg")
        self.assertTrue(output_path.exists())

        rendered = cv2.imread(str(output_path))
        self.assertIsNotNone(rendered)
        self.assertGreater(int(rendered.sum()), 0)

    def test_visualize_detection_file_rejects_missing_image(self):
        detection_path = self.write_detection("scene_0001_view_00.json", self.detection_payload())

        with self.assertRaises(FileNotFoundError):
            self.module.visualize_detection_file(
                detection_path=detection_path,
                image_dir=self.image_dir,
                output_dir=self.output_dir,
                overwrite=False,
            )

    def test_collect_detection_files_sorts_json_files(self):
        self.write_detection("scene_0001_view_01.json", self.detection_payload())
        self.write_detection("notes.txt", {})
        self.write_detection("scene_0001_view_00.json", self.detection_payload())

        files = self.module.collect_detection_files(self.detection_dir)

        self.assertEqual(
            [path.name for path in files],
            ["scene_0001_view_00.json", "scene_0001_view_01.json"],
        )


if __name__ == "__main__":
    unittest.main()
