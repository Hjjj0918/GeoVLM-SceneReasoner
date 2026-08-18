"""Test SAM2 mask overlay visualization generation for manual review.

Usage: python -m pytest tests/test_visualize_masks.py -q
"""

import importlib.util
import json
import os
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_visualize_module():
    module_path = REPO_ROOT / "scripts" / "08_visualize_masks.py"
    spec = importlib.util.spec_from_file_location("visualize_masks", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisualizeMasksTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.image_dir = self.tmp_path / "data" / "images"
        self.mask_dir = self.tmp_path / "outputs" / "masks"
        self.output_dir = self.tmp_path / "outputs" / "visualizations" / "masks"
        self.image_dir.mkdir(parents=True)
        self.mask_dir.mkdir(parents=True)
        self.module = load_visualize_module()
        self.module.REPO_ROOT = self.tmp_path

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_image(self, name: str, width: int = 120, height: int = 80) -> Path:
        path = self.image_dir / name
        image = np.zeros((height, width, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def write_mask(self, segment_dir: Path, name: str = "obj_001.png") -> Path:
        segment_dir.mkdir(parents=True, exist_ok=True)
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[20:60, 30:90] = 255
        path = segment_dir / name
        self.assertTrue(cv2.imwrite(str(path), mask))
        return path

    def write_segments(self, image_stem: str, payload: dict) -> Path:
        segment_dir = self.mask_dir / image_stem
        segment_dir.mkdir(parents=True, exist_ok=True)
        path = segment_dir / "segments.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def segments_payload(self) -> dict:
        return {
            "image": "scene_0001_view_00.jpg",
            "image_width": 120,
            "image_height": 80,
            "source_detection": "outputs/detections_normalized/scene_0001_view_00.json",
            "segmentation_model": "sam2_t.pt",
            "objects": [
                {
                    "object_id": "obj_001",
                    "label": "laptop",
                    "raw_label": "book",
                    "bbox_xyxy": [10, 20, 50, 60],
                    "mask_path": "outputs/masks/scene_0001_view_00/obj_001.png",
                    "mask_area_px": 2400,
                }
            ],
        }

    def test_collect_segment_files_sorts_segments_json_files(self):
        first = self.write_segments("scene_0001_view_01", self.segments_payload())
        second = self.write_segments("scene_0001_view_00", self.segments_payload())
        (self.mask_dir / "notes.txt").write_text("ignore", encoding="utf-8")

        files = self.module.collect_segment_files(self.mask_dir)

        self.assertEqual([path.name for path in files], ["segments.json", "segments.json"])
        self.assertEqual([path.parent.name for path in files], ["scene_0001_view_00", "scene_0001_view_01"])
        self.assertEqual({first, second}, set(files))

    def test_visualize_mask_file_writes_overlay_image(self):
        self.write_image("scene_0001_view_00.jpg")
        self.write_mask(self.mask_dir / "scene_0001_view_00")
        segments_path = self.write_segments("scene_0001_view_00", self.segments_payload())

        output_path = self.module.visualize_mask_file(
            segments_path=segments_path,
            image_dir=self.image_dir,
            output_dir=self.output_dir,
            overwrite=False,
        )

        self.assertEqual(output_path.name, "scene_0001_view_00.jpg")
        self.assertTrue(output_path.exists())

        rendered = cv2.imread(str(output_path))
        self.assertIsNotNone(rendered)
        self.assertGreater(int(rendered.sum()), 0)

    def test_visualize_mask_file_rejects_missing_mask(self):
        self.write_image("scene_0001_view_00.jpg")
        segments_path = self.write_segments("scene_0001_view_00", self.segments_payload())

        with self.assertRaises(FileNotFoundError):
            self.module.visualize_mask_file(
                segments_path=segments_path,
                image_dir=self.image_dir,
                output_dir=self.output_dir,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
