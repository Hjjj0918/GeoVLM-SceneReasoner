"""Test the geometry-only rule baseline and clean-subset question filtering.

Usage: python -m pytest tests/test_run_geometry_rule_baseline.py -q
"""

import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_baseline_module():
    module_path = REPO_ROOT / "scripts" / "12_run_geometry_rule_baseline.py"
    spec = importlib.util.spec_from_file_location("run_geometry_rule_baseline", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeometryRuleBaselineTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.prompt_path = self.tmp_path / "outputs" / "reasoning" / "prompts.jsonl"
        self.geometry_dir = self.tmp_path / "outputs" / "geometry"
        self.output_path = self.tmp_path / "outputs" / "reasoning" / "geometry_rule_baseline.jsonl"
        self.summary_path = self.tmp_path / "outputs" / "evaluations" / "geometry_rule_baseline_summary.json"
        self.geometry_dir.mkdir(parents=True)
        self.module = load_baseline_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def geometry_payload(self) -> dict:
        return {
            "image": "scene_0001_view_00.jpg",
            "depth_order_assumption": "higher_relative_depth_is_closer",
            "objects": [
                {
                    "object_id": "obj_001",
                    "label": "laptop",
                    "mask_area_fraction": 0.2,
                    "relative_depth_median": 5.2,
                    "horizontal_position": "center",
                    "vertical_position": "middle",
                    "depth_order_hint": "near",
                },
                {
                    "object_id": "obj_002",
                    "label": "mouse",
                    "mask_area_fraction": 0.05,
                    "relative_depth_median": 2.1,
                    "horizontal_position": "right",
                    "vertical_position": "middle",
                    "depth_order_hint": "far",
                },
            ],
            "pairwise_relations": [
                {
                    "object_a": "obj_001",
                    "label_a": "laptop",
                    "object_b": "obj_002",
                    "label_b": "mouse",
                    "depth_relation": "closer_than",
                    "horizontal_relation": "left_of",
                    "vertical_relation": "aligned",
                }
            ],
        }

    def prompt_record(self, question_type: str, target_objects: list[str], answer: str = "laptop") -> dict:
        return {
            "question_id": f"{question_type}_q001",
            "image": "scene_0001_view_00.jpg",
            "question": "Which object is closer?",
            "type": question_type,
            "target_objects": target_objects,
            "answer": answer,
            "acceptable_answers": [answer],
            "geometry_path": str(self.geometry_dir / "scene_0001_view_00.json"),
            "missing_target_objects": [],
        }

    def write_geometry(self):
        (self.geometry_dir / "scene_0001_view_00.json").write_text(
            json.dumps(self.geometry_payload()),
            encoding="utf-8",
        )

    def write_prompts(self, records: list[dict]):
        self.prompt_path.parent.mkdir(parents=True)
        with self.prompt_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record) + "\n")

    def test_answer_closer_farther_uses_pairwise_depth_relation(self):
        geometry = self.geometry_payload()
        record = self.prompt_record("closer_farther", ["laptop", "mouse"])

        result = self.module.answer_record(record, geometry)

        self.assertEqual(result["prediction"], "laptop")
        self.assertEqual(result["source"], "pairwise_depth_relation")
        self.assertTrue(result["correct"])

    def test_answer_physical_size_uses_mask_area_fraction(self):
        geometry = self.geometry_payload()
        record = self.prompt_record("physical_size", ["laptop", "mouse"])

        result = self.module.answer_record(record, geometry)

        self.assertEqual(result["prediction"], "laptop")
        self.assertEqual(result["source"], "mask_area_fraction")
        self.assertTrue(result["correct"])

    def test_missing_target_objects_returns_unknown(self):
        geometry = self.geometry_payload()
        record = self.prompt_record("closer_farther", ["laptop", "cup"])
        record["missing_target_objects"] = ["cup"]

        result = self.module.answer_record(record, geometry)

        self.assertEqual(result["prediction"], "unknown")
        self.assertEqual(result["error_reason"], "missing_target_objects")
        self.assertFalse(result["correct"])

    def test_run_baseline_file_writes_results_and_summary(self):
        self.write_geometry()
        missing_record = self.prompt_record("closer_farther", ["laptop", "cup"])
        missing_record["missing_target_objects"] = ["cup"]
        self.write_prompts(
            [
                self.prompt_record("closer_farther", ["laptop", "mouse"]),
                self.prompt_record("physical_size", ["laptop", "mouse"]),
                missing_record,
            ]
        )

        count = self.module.run_baseline_file(
            prompt_path=self.prompt_path,
            output_path=self.output_path,
            summary_path=self.summary_path,
            overwrite=False,
        )

        self.assertEqual(count, 3)
        result_lines = self.output_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(result_lines), 3)
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["answered"], 2)
        self.assertEqual(summary["correct"], 2)
        self.assertEqual(summary["end_to_end_accuracy"], 0.666667)
        self.assertEqual(summary["answered_accuracy"], 1.0)
        self.assertEqual(summary["coverage"], 0.666667)
        self.assertEqual(summary["unknown_rate"], 0.333333)
        self.assertEqual(summary["accuracy"], summary["end_to_end_accuracy"])
        self.assertEqual(summary["by_type"]["closer_farther"]["answered_accuracy"], 1.0)

    def test_run_baseline_file_filters_to_question_ids(self):
        self.write_geometry()
        closer_record = self.prompt_record("closer_farther", ["laptop", "mouse"])
        closer_record["question_id"] = "q_keep"
        size_record = self.prompt_record("physical_size", ["laptop", "mouse"])
        size_record["question_id"] = "q_skip"
        self.write_prompts([closer_record, size_record])

        count = self.module.run_baseline_file(
            prompt_path=self.prompt_path,
            output_path=self.output_path,
            summary_path=self.summary_path,
            overwrite=False,
            question_ids={"q_keep"},
        )

        self.assertEqual(count, 1)
        result_lines = self.output_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(result_lines), 1)
        result = json.loads(result_lines[0])
        self.assertEqual(result["question_id"], "q_keep")
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["correct"], 1)

    def test_run_baseline_file_rejects_existing_output_without_overwrite(self):
        self.write_geometry()
        self.write_prompts([self.prompt_record("closer_farther", ["laptop", "mouse"])])
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.module.run_baseline_file(
                prompt_path=self.prompt_path,
                output_path=self.output_path,
                summary_path=self.summary_path,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
