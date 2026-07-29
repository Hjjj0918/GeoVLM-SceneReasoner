import importlib.util
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_check_dataset_module():
    module_path = REPO_ROOT / "scripts" / "00_check_dataset.py"
    spec = importlib.util.spec_from_file_location("check_dataset", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_image(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


class DatasetCheckTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.module = load_check_dataset_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_collect_image_records_reports_dimensions_and_recommended_names(self):
        image_dir = self.tmp_path / "data" / "images"
        write_image(image_dir / "scene_0001.jpg", width=32, height=24)
        write_image(image_dir / "bad_name.png", width=40, height=30)

        records = self.module.collect_image_records(image_dir, expected_prefix="scene")

        self.assertEqual([record.path.name for record in records], ["bad_name.png", "scene_0001.jpg"])
        self.assertEqual(records[0].width, 40)
        self.assertEqual(records[0].height, 30)
        self.assertFalse(records[0].has_recommended_name)
        self.assertTrue(records[1].has_recommended_name)

    def test_scene_view_names_are_recommended_names(self):
        image_dir = self.tmp_path / "data" / "images"
        write_image(image_dir / "scene_0001_view_00.jpg", width=32, height=24)

        records = self.module.collect_image_records(image_dir, expected_prefix="scene")

        self.assertEqual(records[0].path.name, "scene_0001_view_00.jpg")
        self.assertTrue(records[0].has_recommended_name)

    def test_summarize_dimensions_detects_inconsistent_sizes(self):
        image_dir = self.tmp_path / "data" / "images"
        write_image(image_dir / "scene_0001.jpg", width=32, height=24)
        write_image(image_dir / "scene_0002.jpg", width=64, height=48)

        records = self.module.collect_image_records(image_dir, expected_prefix="scene")
        summary = self.module.summarize_dimensions(records)

        self.assertEqual(summary.total_images, 2)
        self.assertEqual(summary.unique_dimensions, [(32, 24), (64, 48)])
        self.assertTrue(summary.has_inconsistent_dimensions)

    def test_empty_directory_is_valid_dataset_state(self):
        image_dir = self.tmp_path / "data" / "images"
        image_dir.mkdir(parents=True)

        records = self.module.collect_image_records(image_dir, expected_prefix="scene")
        summary = self.module.summarize_dimensions(records)

        self.assertEqual(records, [])
        self.assertEqual(summary.total_images, 0)
        self.assertFalse(summary.has_inconsistent_dimensions)

    def test_dataset_status_reports_missing_question_file(self):
        image_dir = self.tmp_path / "data" / "images"
        question_path = self.tmp_path / "data" / "questions.json"
        image_dir.mkdir(parents=True)

        status = self.module.build_dataset_status(
            scene_dir=image_dir,
            question_path=question_path,
        )

        self.assertEqual(status.scene_summary.total_images, 0)
        self.assertFalse(status.question_file_exists)
        self.assertEqual(status.question_count, 0)

    def test_dataset_status_counts_questions_when_file_exists(self):
        image_dir = self.tmp_path / "data" / "images"
        question_path = self.tmp_path / "data" / "questions.json"
        image_dir.mkdir(parents=True)
        question_path.write_text(
            '{"questions": [{"question_id": "q001"}, {"question_id": "q002"}]}',
            encoding="utf-8",
        )

        status = self.module.build_dataset_status(
            scene_dir=image_dir,
            question_path=question_path,
        )

        self.assertTrue(status.question_file_exists)
        self.assertEqual(status.question_count, 2)


if __name__ == "__main__":
    unittest.main()
