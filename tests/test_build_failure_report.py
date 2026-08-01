import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_report_module():
    module_path = REPO_ROOT / "scripts" / "13_build_failure_report.py"
    spec = importlib.util.spec_from_file_location("build_failure_report", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildFailureReportTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.baseline_path = self.tmp_path / "outputs" / "reasoning" / "geometry_rule_baseline.jsonl"
        self.report_path = self.tmp_path / "outputs" / "evaluations" / "pipeline_failure_report.json"
        self.csv_path = self.tmp_path / "outputs" / "evaluations" / "pipeline_failure_report.csv"
        self.module = load_report_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_baseline(self, records: list[dict]):
        self.baseline_path.parent.mkdir(parents=True)
        with self.baseline_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record) + "\n")

    def missing_record(self, question_id: str, image: str, missing: list[str]) -> dict:
        return {
            "question_id": question_id,
            "image": image,
            "type": "closer_farther",
            "target_objects": ["laptop", "mouse"],
            "prediction": "unknown",
            "correct": False,
            "error_reason": "missing_target_objects",
            "missing_target_objects": missing,
        }

    def test_build_failure_report_groups_missing_targets_by_image_and_label(self):
        records = [
            self.missing_record("q001", "scene_0001_view_00.jpg", ["laptop", "mouse"]),
            self.missing_record("q002", "scene_0001_view_00.jpg", ["laptop"]),
            self.missing_record("q003", "scene_0001_view_01.jpg", ["cup"]),
            {"question_id": "q004", "image": "scene_0001_view_01.jpg", "prediction": "laptop", "correct": True},
        ]

        report = self.module.build_failure_report(records)

        self.assertEqual(report["total_records"], 4)
        self.assertEqual(report["missing_target_records"], 3)
        self.assertEqual(report["missing_target_rate"], 0.75)
        self.assertEqual(report["missing_label_counts"], {"laptop": 2, "mouse": 1, "cup": 1})
        self.assertEqual(report["images"][0]["image"], "scene_0001_view_00.jpg")
        self.assertEqual(report["images"][0]["missing_label_counts"], {"laptop": 2, "mouse": 1})

    def test_write_failure_report_writes_json_and_csv(self):
        self.write_baseline(
            [
                self.missing_record("q001", "scene_0001_view_00.jpg", ["laptop", "mouse"]),
                {"question_id": "q002", "image": "scene_0001_view_01.jpg", "prediction": "laptop", "correct": True},
            ]
        )

        count = self.module.write_failure_report(
            baseline_path=self.baseline_path,
            report_path=self.report_path,
            csv_path=self.csv_path,
            overwrite=False,
        )

        self.assertEqual(count, 2)
        self.assertTrue(self.report_path.exists())
        self.assertTrue(self.csv_path.exists())
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["missing_target_records"], 1)
        csv_text = self.csv_path.read_text(encoding="utf-8")
        self.assertIn("scene_0001_view_00.jpg", csv_text)
        self.assertIn("laptop;mouse", csv_text)

    def test_write_failure_report_rejects_existing_outputs_without_overwrite(self):
        self.write_baseline([])
        self.report_path.parent.mkdir(parents=True)
        self.report_path.write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.module.write_failure_report(
                baseline_path=self.baseline_path,
                report_path=self.report_path,
                csv_path=self.csv_path,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
