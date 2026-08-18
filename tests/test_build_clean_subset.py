"""Test clean subset generation from manual evaluation review CSV files.

Usage: python -m pytest tests/test_build_clean_subset.py -q
"""

import csv
import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_clean_subset_module():
    module_path = REPO_ROOT / "scripts" / "15_build_clean_subset.py"
    spec = importlib.util.spec_from_file_location("build_clean_subset", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildCleanSubsetTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.review_path = self.tmp_path / "outputs" / "evaluations" / "evaluation_review.csv"
        self.questions_path = self.tmp_path / "data" / "questions.json"
        self.output_path = self.tmp_path / "outputs" / "evaluations" / "clean_subset.json"
        self.questions_output_path = self.tmp_path / "outputs" / "evaluations" / "clean_subset_questions.json"
        self.module = load_clean_subset_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_review(self, rows: list[dict[str, str]]):
        self.review_path.parent.mkdir(parents=True)
        fieldnames = [
            "question_id",
            "image",
            "type",
            "target_objects",
            "automatic_split",
            "mask_ok",
            "depth_ok",
            "question_valid",
            "final_split",
            "review_notes",
        ]
        with self.review_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_questions(self):
        self.questions_path.parent.mkdir(parents=True)
        self.questions_path.write_text(
            json.dumps(
                {
                    "version": "0.1",
                    "scene_group": "scene_0001",
                    "questions": [
                        {"question_id": "q001", "image": "scene_0001_view_00.jpg", "question": "A?"},
                        {"question_id": "q002", "image": "scene_0001_view_00.jpg", "question": "B?"},
                        {"question_id": "q003", "image": "scene_0001_view_01.jpg", "question": "C?"},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def row(
        self,
        question_id: str,
        automatic_split: str = "geometry_available_candidates",
        mask_ok: str = "yes",
        depth_ok: str = "yes",
        question_valid: str = "yes",
        final_split: str = "",
    ) -> dict[str, str]:
        return {
            "question_id": question_id,
            "image": "scene_0001_view_00.jpg",
            "type": "closer_farther",
            "target_objects": '["laptop","mouse"]',
            "automatic_split": automatic_split,
            "mask_ok": mask_ok,
            "depth_ok": depth_ok,
            "question_valid": question_valid,
            "final_split": final_split,
            "review_notes": "",
        }

    def test_build_clean_subset_accepts_only_reviewed_clean_candidates(self):
        rows = [
            self.row("q001"),
            self.row("q002", mask_ok="no"),
            self.row("q003", automatic_split="pipeline_failure"),
        ]

        payload = self.module.build_clean_subset_payload(rows, source="review.csv")

        self.assertEqual(payload["counts"]["review_rows"], 3)
        self.assertEqual(payload["counts"]["clean_subset"], 1)
        self.assertEqual(payload["counts"]["rejected_or_pending"], 2)
        self.assertEqual(payload["question_ids"], ["q001"])

    def test_write_clean_subset_files_filters_questions_file(self):
        self.write_review([self.row("q001"), self.row("q002", depth_ok="pending")])
        self.write_questions()

        count = self.module.write_clean_subset_files(
            review_path=self.review_path,
            questions_path=self.questions_path,
            output_path=self.output_path,
            questions_output_path=self.questions_output_path,
            overwrite=False,
        )

        self.assertEqual(count, 1)
        payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        questions_payload = json.loads(self.questions_output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["question_ids"], ["q001"])
        self.assertEqual([question["question_id"] for question in questions_payload["questions"]], ["q001"])

    def test_write_clean_subset_files_rejects_existing_outputs_without_overwrite(self):
        self.write_review([])
        self.write_questions()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.module.write_clean_subset_files(
                review_path=self.review_path,
                questions_path=self.questions_path,
                output_path=self.output_path,
                questions_output_path=self.questions_output_path,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
