"""Test SAM2 segmentation outputs generated from normalized detection boxes.

Usage: python -m pytest tests/test_segment_objects.py -q
"""

import importlib.util
import json
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_segment_module():
    module_path = REPO_ROOT / "scripts" / "07_segment_objects.py"
    spec = importlib.util.spec_from_file_location("segment_objects", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SegmentObjectsTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.image_dir = self.tmp_path / "data" / "images"
        self.detection_dir = self.tmp_path / "outputs" / "detections_normalized"
        self.output_dir = self.tmp_path / "outputs" / "masks"
        self.image_dir.mkdir(parents=True)
        self.detection_dir.mkdir(parents=True)
        self.module = load_segment_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_image(self, name: str, width: int = 64, height: int = 48) -> Path:
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
            "image_width": 64,
            "image_height": 48,
            "model": "yolo11n.pt",
            "objects": [
                {
                    "object_id": "obj_001",
                    "label": "laptop",
                    "raw_label": "book",
                    "confidence": 0.5,
                    "bbox_xyxy": [10, 12, 40, 32],
                }
            ],
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

    def test_prepare_box_prompt_uses_bbox_xyxy(self):
        detection = self.detection_payload()["objects"][0]

        box = self.module.prepare_box_prompt(detection)

        self.assertEqual(box, [10.0, 12.0, 40.0, 32.0])

    def test_mask_to_uint8_resizes_and_binarizes(self):
        mask = np.array([[0.0, 0.6], [0.8, 0.2]], dtype=np.float32)

        rendered = self.module.mask_to_uint8(mask, width=4, height=4)

        self.assertEqual(rendered.shape, (4, 4))
        self.assertEqual(set(np.unique(rendered).tolist()), {0, 255})

    def test_build_segment_record_preserves_detection_metadata(self):
        detection = self.detection_payload()["objects"][0]

        record = self.module.build_segment_record(
            detection=detection,
            mask_path=Path("outputs/masks/scene_0001_view_00/obj_001.png"),
            mask_area_px=42,
        )

        self.assertEqual(record["object_id"], "obj_001")
        self.assertEqual(record["label"], "laptop")
        self.assertEqual(record["raw_label"], "book")
        self.assertEqual(record["mask_area_px"], 42)

    def test_write_segments_json_writes_expected_payload(self):
        segment_dir = self.output_dir / "scene_0001_view_00"
        records = [
            {
                "object_id": "obj_001",
                "label": "laptop",
                "bbox_xyxy": [10, 12, 40, 32],
                "mask_path": "outputs/masks/scene_0001_view_00/obj_001.png",
                "mask_area_px": 42,
            }
        ]

        output_path = self.module.write_segments_json(
            segment_dir=segment_dir,
            image_name="scene_0001_view_00.jpg",
            image_width=64,
            image_height=48,
            source_detection="outputs/detections_normalized/scene_0001_view_00.json",
            records=records,
            model_name="sam2_t.pt",
            overwrite=False,
        )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["image"], "scene_0001_view_00.jpg")
        self.assertEqual(payload["segmentation_model"], "sam2_t.pt")
        self.assertEqual(payload["objects"], records)

    def test_segment_detection_file_with_empty_objects_writes_empty_segments(self):
        self.write_image("scene_0001_view_00.jpg")
        detection_path = self.write_detection(
            "scene_0001_view_00.json",
            {
                "image": "scene_0001_view_00.jpg",
                "image_width": 64,
                "image_height": 48,
                "objects": [],
            },
        )

        output_path = self.module.segment_detection_file(
            model=None,
            detection_path=detection_path,
            image_dir=self.image_dir,
            output_dir=self.output_dir,
            model_name="sam2_t.pt",
            overwrite=False,
        )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["objects"], [])


if __name__ == "__main__":
    unittest.main()
