import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_split_module():
    module_path = REPO_ROOT / "scripts" / "14_build_evaluation_splits.py"
    spec = importlib.util.spec_from_file_location("build_evaluation_splits", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildEvaluationSplitsTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.baseline_path = self.tmp_path / "outputs" / "reasoning" / "geometry_rule_baseline.jsonl"
        self.split_path = self.tmp_path / "outputs" / "evaluations" / "evaluation_splits.json"
        self.review_path = self.tmp_path / "outputs" / "evaluations" / "evaluation_review.csv"
        self.module = load_split_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def record(self, question_id: str, image: str, missing: list[str] | None = None) -> dict:
        return {
            "question_id": question_id,
            "image": image,
            "type": "closer_farther",
            "target_objects": ["laptop", "mouse"],
            "prediction": "unknown" if missing else "laptop",
            "correct": not bool(missing),
            "error_reason": "missing_target_objects" if missing else None,
            "missing_target_objects": missing or [],
        }

    def write_records(self, records: list[dict]):
        self.baseline_path.parent.mkdir(parents=True)
        with self.baseline_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record) + "\n")

    def test_build_split_payload_separates_pipeline_failures_from_candidates(self):
        records = [
            self.record("q001", "scene_0001_view_00.jpg"),
            self.record("q002", "scene_0001_view_00.jpg", ["laptop"]),
            self.record("q003", "scene_0001_view_01.jpg"),
        ]

        payload = self.module.build_split_payload(records)

        self.assertEqual(payload["counts"]["total"], 3)
        self.assertEqual(payload["counts"]["geometry_available_candidates"], 2)
        self.assertEqual(payload["counts"]["pipeline_failure"], 1)
        self.assertEqual(payload["counts"]["manual_review_pending"], 2)
        self.assertEqual(payload["splits"]["pipeline_failure"], ["q002"])
        self.assertEqual(payload["review_status"], "geometry_available_candidates require manual mask/depth review")

    def test_write_split_files_writes_json_and_review_csv(self):
        self.write_records(
            [
                self.record("q001", "scene_0001_view_00.jpg"),
                self.record("q002", "scene_0001_view_00.jpg", ["laptop"]),
            ]
        )

        count = self.module.write_split_files(
            baseline_path=self.baseline_path,
            split_path=self.split_path,
            review_path=self.review_path,
            overwrite=False,
        )

        self.assertEqual(count, 2)
        self.assertTrue(self.split_path.exists())
        self.assertTrue(self.review_path.exists())
        payload = json.loads(self.split_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["counts"]["pipeline_failure"], 1)
        csv_text = self.review_path.read_text(encoding="utf-8")
        self.assertIn("q002", csv_text)
        self.assertIn("pipeline_failure", csv_text)
        self.assertIn("mask_ok", csv_text)
        self.assertIn("depth_ok", csv_text)
        self.assertIn("question_valid", csv_text)
        self.assertIn("final_split", csv_text)
        self.assertIn("review_notes", csv_text)

    def test_write_split_files_rejects_existing_output_without_overwrite(self):
        self.write_records([])
        self.split_path.parent.mkdir(parents=True)
        self.split_path.write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.module.write_split_files(
                baseline_path=self.baseline_path,
                split_path=self.split_path,
                review_path=self.review_path,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
