"""Test object-level geometry extraction from masks and relative depth maps.

Usage: python -m pytest tests/test_extract_geometry.py -q
"""

import importlib.util
import json
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_geometry_module():
    module_path = REPO_ROOT / "scripts" / "10_extract_geometry.py"
    spec = importlib.util.spec_from_file_location("extract_geometry", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExtractGeometryTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.mask_dir = self.tmp_path / "outputs" / "masks"
        self.depth_dir = self.tmp_path / "outputs" / "depth"
        self.output_dir = self.tmp_path / "outputs" / "geometry"
        self.mask_dir.mkdir(parents=True)
        self.depth_dir.mkdir(parents=True)
        self.module = load_geometry_module()
        self.module.REPO_ROOT = self.tmp_path

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_mask(self, image_stem: str, object_id: str, left: int, top: int, right: int, bottom: int) -> Path:
        segment_dir = self.mask_dir / image_stem
        segment_dir.mkdir(parents=True, exist_ok=True)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[top:bottom, left:right] = 255
        path = segment_dir / f"{object_id}.png"
        self.assertTrue(cv2.imwrite(str(path), mask))
        return path

    def write_segments(self, image_stem: str) -> Path:
        self.write_mask(image_stem, "obj_001", 1, 2, 4, 6)
        self.write_mask(image_stem, "obj_002", 6, 1, 9, 5)
        payload = {
            "image": f"{image_stem}.jpg",
            "image_width": 10,
            "image_height": 10,
            "source_detection": f"outputs/detections_normalized/{image_stem}.json",
            "segmentation_model": "sam2_t.pt",
            "objects": [
                {
                    "object_id": "obj_001",
                    "label": "cup",
                    "confidence": 0.9,
                    "bbox_xyxy": [1, 2, 4, 6],
                    "mask_path": f"outputs/masks/{image_stem}/obj_001.png",
                    "mask_area_px": 12,
                },
                {
                    "object_id": "obj_002",
                    "label": "laptop",
                    "confidence": 0.8,
                    "bbox_xyxy": [6, 1, 9, 5],
                    "mask_path": f"outputs/masks/{image_stem}/obj_002.png",
                    "mask_area_px": 12,
                },
            ],
        }
        path = self.mask_dir / image_stem / "segments.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def write_depth(self, image_stem: str) -> Path:
        depth = np.ones((10, 10), dtype=np.float32)
        depth[:, 0:5] = 2.0
        depth[:, 5:10] = 8.0
        path = self.depth_dir / f"{image_stem}.npy"
        np.save(path, depth)
        return path

    def test_collect_segment_files_sorts_segments_json_files(self):
        self.write_segments("scene_0001_view_01")
        self.write_segments("scene_0001_view_00")

        files = self.module.collect_segment_files(self.mask_dir)

        self.assertEqual([path.parent.name for path in files], ["scene_0001_view_00", "scene_0001_view_01"])

    def test_mask_geometry_computes_centroid_bounds_and_area_fraction(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 1:4] = True

        geometry = self.module.compute_mask_geometry(mask, image_width=10, image_height=10)

        self.assertEqual(geometry["mask_area_px"], 12)
        self.assertEqual(geometry["mask_bounds_xyxy"], [1, 2, 3, 5])
        self.assertEqual(geometry["mask_centroid"], [2.0, 3.5])
        self.assertEqual(geometry["mask_area_fraction"], 0.12)

    def test_depth_stats_for_mask_computes_summary_values(self):
        depth = np.arange(100, dtype=np.float32).reshape(10, 10)
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True
        mask[0, 1] = True
        mask[0, 2] = True

        stats = self.module.compute_depth_stats(depth, mask)

        self.assertEqual(stats["relative_depth_min"], 0.0)
        self.assertEqual(stats["relative_depth_max"], 2.0)
        self.assertEqual(stats["relative_depth_mean"], 1.0)
        self.assertEqual(stats["relative_depth_median"], 1.0)

    def test_build_pairwise_relations_uses_centers_and_depth(self):
        objects = [
            {"object_id": "obj_001", "label": "cup", "mask_centroid": [2.0, 3.5], "relative_depth_median": 2.0},
            {"object_id": "obj_002", "label": "laptop", "mask_centroid": [7.0, 3.0], "relative_depth_median": 8.0},
        ]

        relations = self.module.build_pairwise_relations(
            objects,
            image_width=10,
            image_height=10,
            depth_range=7.0,
            higher_depth_is_closer=True,
        )

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["object_a"], "obj_001")
        self.assertEqual(relations[0]["object_b"], "obj_002")
        self.assertEqual(relations[0]["horizontal_relation"], "left_of")
        self.assertEqual(relations[0]["depth_relation"], "farther_than")

    def test_extract_geometry_file_writes_expected_payload(self):
        image_stem = "scene_0001_view_00"
        segments_path = self.write_segments(image_stem)
        self.write_depth(image_stem)

        output_path = self.module.extract_geometry_file(
            segments_path=segments_path,
            depth_dir=self.depth_dir,
            output_dir=self.output_dir,
            overwrite=False,
            higher_depth_is_closer=True,
        )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["image"], f"{image_stem}.jpg")
        self.assertEqual(payload["depth_order_assumption"], "higher_relative_depth_is_closer")
        self.assertEqual(len(payload["objects"]), 2)
        self.assertEqual(payload["objects"][0]["label"], "cup")
        self.assertEqual(payload["objects"][0]["horizontal_position"], "left")
        self.assertEqual(payload["objects"][1]["depth_order_hint"], "near")
        self.assertEqual(payload["pairwise_relations"][0]["depth_relation"], "farther_than")


if __name__ == "__main__":
    unittest.main()
