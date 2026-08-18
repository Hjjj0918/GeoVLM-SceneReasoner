"""Test YOLO detection output formatting and detection script control flow.

Usage: python -m pytest tests/test_detect_objects.py -q
"""

import importlib.util
import json
import os
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_detect_module():
    module_path = REPO_ROOT / "scripts" / "04_detect_objects.py"
    spec = importlib.util.spec_from_file_location("detect_objects", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DetectObjectsTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.image_dir = self.tmp_path / "data" / "images"
        self.image_dir.mkdir(parents=True)
        self.module = load_detect_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def touch(self, name: str) -> Path:
        path = self.image_dir / name
        path.write_bytes(b"image")
        return path

    def test_collect_image_files_sorts_images_and_ignores_non_images(self):
        self.touch("scene_0001_view_01.jpg")
        self.touch("notes.txt")
        self.touch("scene_0001_view_00.jpg")

        files = self.module.collect_image_files(self.image_dir)

        self.assertEqual(
            [path.name for path in files],
            ["scene_0001_view_00.jpg", "scene_0001_view_01.jpg"],
        )

    def test_build_detection_payload_assigns_stable_object_ids(self):
        records = [
            self.module.DetectionRecord(
                label="laptop",
                confidence=0.91,
                bbox_xyxy=[10.0, 20.0, 30.0, 40.0],
            ),
            self.module.DetectionRecord(
                label="mouse",
                confidence=0.82,
                bbox_xyxy=[50.0, 60.0, 70.0, 80.0],
            ),
        ]
        image_path = self.image_dir / "scene_0001_view_00.jpg"

        payload = self.module.build_detection_payload(
            image_path=image_path,
            image_width=1280,
            image_height=720,
            detections=records,
            model_name="yolo11n.pt",
        )

        self.assertEqual(payload["image"], "scene_0001_view_00.jpg")
        self.assertEqual(payload["image_width"], 1280)
        self.assertEqual(payload["image_height"], 720)
        self.assertEqual(payload["model"], "yolo11n.pt")
        self.assertEqual(payload["objects"][0]["object_id"], "obj_001")
        self.assertEqual(payload["objects"][0]["label"], "laptop")
        self.assertEqual(payload["objects"][1]["object_id"], "obj_002")

    def test_extract_yolo_detections_converts_result_boxes(self):
        class FakeBoxes:
            xyxy = [[10, 20, 30, 40], [50, 60, 70, 80]]
            conf = [0.91, 0.32]
            cls = [0, 1]

        class FakeResult:
            boxes = FakeBoxes()
            names = {0: "laptop", 1: "mouse"}

        detections = self.module.extract_yolo_detections(FakeResult(), min_confidence=0.5)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "laptop")
        self.assertEqual(detections[0].confidence, 0.91)
        self.assertEqual(detections[0].bbox_xyxy, [10.0, 20.0, 30.0, 40.0])

    def test_prepare_ultralytics_environment_sets_config_dir(self):
        previous_value = os.environ.pop("YOLO_CONFIG_DIR", None)
        config_dir = self.tmp_path / "outputs" / "ultralytics_config"
        try:
            self.module.prepare_ultralytics_environment(config_dir)

            self.assertEqual(os.environ["YOLO_CONFIG_DIR"], str(config_dir.resolve()))
            self.assertTrue(config_dir.exists())
        finally:
            if previous_value is not None:
                os.environ["YOLO_CONFIG_DIR"] = previous_value
            else:
                os.environ.pop("YOLO_CONFIG_DIR", None)

    def test_write_detection_payload_rejects_existing_output_without_overwrite(self):
        output_path = self.tmp_path / "outputs" / "detections" / "scene_0001_view_00.json"
        output_path.parent.mkdir(parents=True)
        output_path.write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.module.write_detection_payload({"objects": []}, output_path, overwrite=False)

    def test_write_detection_payload_allows_overwrite(self):
        output_path = self.tmp_path / "outputs" / "detections" / "scene_0001_view_00.json"
        output_path.parent.mkdir(parents=True)
        output_path.write_text("{}", encoding="utf-8")

        self.module.write_detection_payload({"objects": []}, output_path, overwrite=True)

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["objects"], [])


if __name__ == "__main__":
    unittest.main()
