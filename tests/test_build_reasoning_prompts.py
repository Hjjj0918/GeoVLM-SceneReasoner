"""Test reasoning prompt generation for Pure VLM, Geometry-only LLM, and GeoVLM.

Usage: python -m pytest tests/test_build_reasoning_prompts.py -q
"""

import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_prompt_module():
    module_path = REPO_ROOT / "scripts" / "11_build_reasoning_prompts.py"
    spec = importlib.util.spec_from_file_location("build_reasoning_prompts", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildReasoningPromptsTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.question_path = self.tmp_path / "data" / "questions.json"
        self.geometry_dir = self.tmp_path / "outputs" / "geometry"
        self.output_path = self.tmp_path / "outputs" / "reasoning" / "prompts.jsonl"
        self.question_path.parent.mkdir(parents=True)
        self.geometry_dir.mkdir(parents=True)
        self.module = load_prompt_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def question_payload(self) -> dict:
        return {
            "version": "0.1",
            "questions": [
                {
                    "question_id": "scene_0001_view_00_q001",
                    "image": "scene_0001_view_00.jpg",
                    "question": "Which object is closer to the camera, laptop or mouse?",
                    "type": "closer_farther",
                    "target_objects": ["laptop", "mouse"],
                    "answer": "laptop",
                    "evaluation": {"metric": "exact_match", "acceptable_answers": ["laptop"]},
                }
            ],
        }

    def geometry_payload(self) -> dict:
        return {
            "image": "scene_0001_view_00.jpg",
            "image_width": 100,
            "image_height": 80,
            "depth_order_assumption": "higher_relative_depth_is_closer",
            "objects": [
                {
                    "object_id": "obj_001",
                    "label": "laptop",
                    "mask_centroid": [30.0, 50.0],
                    "mask_area_fraction": 0.2,
                    "relative_depth_median": 5.2,
                    "closeness_score": 5.8,
                    "relative_depth_percentile": 0.8,
                    "horizontal_position": "left",
                    "vertical_position": "middle",
                    "depth_order_hint": "near",
                },
                {
                    "object_id": "obj_002",
                    "label": "mouse",
                    "mask_centroid": [70.0, 55.0],
                    "mask_area_fraction": 0.05,
                    "relative_depth_median": 2.1,
                    "closeness_score": 2.6,
                    "relative_depth_percentile": 0.3,
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
                    "horizontal_relation": "left_of",
                    "vertical_relation": "aligned",
                    "depth_relation": "closer_than",
                }
            ],
        }

    def write_inputs(self):
        self.question_path.write_text(json.dumps(self.question_payload()), encoding="utf-8")
        (self.geometry_dir / "scene_0001_view_00.json").write_text(
            json.dumps(self.geometry_payload()),
            encoding="utf-8",
        )

    def test_geometry_path_for_image_uses_image_stem(self):
        path = self.module.geometry_path_for_image(self.geometry_dir, "scene_0001_view_00.jpg")

        self.assertEqual(path, self.geometry_dir / "scene_0001_view_00.json")

    def test_summarize_geometry_includes_objects_and_relations(self):
        question = self.question_payload()["questions"][0]
        summary = self.module.summarize_geometry(
            self.geometry_payload(),
            target_objects=["laptop", "mouse"],
            question_type=question["type"],
        )

        self.assertIn("Closeness score: higher means closer", summary)
        self.assertIn("Closer/farther standard: compare the nearest visible surface", summary)
        self.assertIn("obj_001 laptop", summary)
        self.assertIn("position=left/middle", summary)
        self.assertIn("closeness_score=5.8", summary)
        self.assertNotIn("median_depth=", summary)
        self.assertIn("Estimated closer object from geometry: laptop", summary)
        self.assertIn("obj_001 laptop vs obj_002 mouse: horizontal=left_of", summary)

    def test_build_prompt_record_omits_answer_from_prompt_text(self):
        question = self.question_payload()["questions"][0]
        record = self.module.build_prompt_record(
            question=question,
            geometry=self.geometry_payload(),
            geometry_path=self.geometry_dir / "scene_0001_view_00.json",
        )

        self.assertEqual(record["question_id"], "scene_0001_view_00_q001")
        self.assertEqual(record["answer"], "laptop")
        self.assertEqual(record["missing_target_objects"], [])
        self.assertIn("pure_vlm_prompt", record)
        self.assertIn("geometry_llm_prompt", record)
        self.assertIn("geovlm_prompt", record)
        self.assertNotIn("Answer: laptop", record["geometry_llm_prompt"])

    def test_build_prompts_file_writes_one_jsonl_record(self):
        self.write_inputs()

        written = self.module.build_prompts_file(
            question_path=self.question_path,
            geometry_dir=self.geometry_dir,
            output_path=self.output_path,
            overwrite=False,
        )

        self.assertEqual(written, 1)
        lines = self.output_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["image"], "scene_0001_view_00.jpg")

    def test_build_prompts_file_rejects_existing_output_without_overwrite(self):
        self.write_inputs()
        self.output_path.parent.mkdir(parents=True)
        self.output_path.write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.module.build_prompts_file(
                question_path=self.question_path,
                geometry_dir=self.geometry_dir,
                output_path=self.output_path,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
