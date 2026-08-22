"""Test model-agnostic VLM inference orchestration and answer normalization.

Usage: python -m pytest tests/test_run_vlm_inference.py -q
"""

import importlib.util
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_inference_module():
    module_path = REPO_ROOT / "scripts" / "16_run_vlm_inference.py"
    spec = importlib.util.spec_from_file_location("run_vlm_inference", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProvider:
    model_name = "fake-vlm"

    def __init__(self, response: str = "laptop"):
        self.response = response
        self.calls = []

    def answer(self, prompt: str, image_path: Path | None = None) -> str:
        self.calls.append({"prompt": prompt, "image_path": image_path})
        return self.response


class RunVlmInferenceTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.prompt_path = self.tmp_path / "outputs" / "reasoning" / "prompts.jsonl"
        self.clean_subset_path = self.tmp_path / "outputs" / "evaluations" / "clean_subset.json"
        self.output_path = self.tmp_path / "outputs" / "inference" / "pure_vlm.jsonl"
        self.summary_path = self.tmp_path / "outputs" / "evaluations" / "pure_vlm_summary.json"
        self.images_dir = self.tmp_path / "data" / "images"
        self.images_dir.mkdir(parents=True)
        self.image_path = self.images_dir / "scene_0001_view_01.jpg"
        self.image_path.write_bytes(b"fake-image")
        self.module = load_inference_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def record(self, question_id: str, answer: str = "laptop") -> dict:
        return {
            "question_id": question_id,
            "image": "scene_0001_view_01.jpg",
            "question": "Which object is closer?",
            "type": "closer_farther",
            "target_objects": ["laptop", "mouse"],
            "answer": answer,
            "acceptable_answers": [answer],
            "pure_vlm_prompt": "Use the image. Answer with one label.",
            "geometry_llm_prompt": "Use the geometry. Answer with one label.",
            "geovlm_prompt": "Use image and geometry. Answer with one label.",
        }

    def write_inputs(self):
        self.prompt_path.parent.mkdir(parents=True)
        with self.prompt_path.open("w", encoding="utf-8") as file:
            file.write(json.dumps(self.record("q001")) + "\n")
            file.write(json.dumps(self.record("q002")) + "\n")
        self.clean_subset_path.parent.mkdir(parents=True)
        self.clean_subset_path.write_text(
            json.dumps({"question_ids": ["q001"]}),
            encoding="utf-8",
        )

    def test_normalize_prediction_extracts_target_label_from_model_sentence(self):
        prediction = self.module.normalize_prediction(
            "The laptop is closer to the camera.",
            ["laptop", "mouse"],
        )

        self.assertEqual(prediction, "laptop")
        self.assertEqual(self.module.normalize_prediction("unknown", ["laptop"]), "unknown")

    def test_select_prompt_uses_track_specific_prompt(self):
        record = self.record("q001")

        self.assertEqual(self.module.select_prompt(record, "pure_vlm"), record["pure_vlm_prompt"])
        self.assertEqual(self.module.select_prompt(record, "geometry_only"), record["geometry_llm_prompt"])
        self.assertEqual(self.module.select_prompt(record, "geovlm"), record["geovlm_prompt"])

    def test_run_inference_filters_to_clean_question_ids_and_writes_summary(self):
        self.write_inputs()
        provider = FakeProvider("The laptop is closer.")

        count = self.module.run_inference_file(
            prompt_path=self.prompt_path,
            clean_subset_path=self.clean_subset_path,
            images_dir=self.images_dir,
            output_path=self.output_path,
            summary_path=self.summary_path,
            track="pure_vlm",
            provider=provider,
            overwrite=False,
        )

        self.assertEqual(count, 1)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0]["image_path"], self.image_path)
        result = json.loads(self.output_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(result["question_id"], "q001")
        self.assertEqual(result["prediction"], "laptop")
        self.assertTrue(result["correct"])
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["correct"], 1)

    def test_run_inference_prints_progress_after_each_question(self):
        self.write_inputs()
        provider = FakeProvider("The laptop is closer.")
        progress_stream = io.StringIO()

        self.module.run_inference_file(
            prompt_path=self.prompt_path,
            clean_subset_path=self.clean_subset_path,
            images_dir=self.images_dir,
            output_path=self.output_path,
            summary_path=self.summary_path,
            track="pure_vlm",
            provider=provider,
            overwrite=False,
            progress_stream=progress_stream,
        )

        output = progress_stream.getvalue()
        self.assertIn("[1/1]", output)
        self.assertIn("q001", output)
        self.assertIn("closer_farther", output)
        self.assertIn("-> laptop", output)
        self.assertIn("correct=True", output)

    def test_run_inference_rejects_existing_outputs_without_overwrite(self):
        self.write_inputs()
        self.output_path.parent.mkdir(parents=True)
        self.output_path.write_text("", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.module.run_inference_file(
                prompt_path=self.prompt_path,
                clean_subset_path=self.clean_subset_path,
                images_dir=self.images_dir,
                output_path=self.output_path,
                summary_path=self.summary_path,
                track="pure_vlm",
                provider=FakeProvider(),
                overwrite=False,
            )

    def test_cli_help_runs_when_script_is_invoked_directly(self):
        result = subprocess.run(
            [sys.executable, "scripts/16_run_vlm_inference.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
