"""Test benchmark question schema validation and error reporting.

Usage: python -m pytest tests/test_validate_questions.py -q
"""

import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_validator_module():
    module_path = REPO_ROOT / "scripts" / "01_validate_questions.py"
    spec = importlib.util.spec_from_file_location("validate_questions", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuestionValidatorTest(unittest.TestCase):
    def setUp(self):
        self.module = load_validator_module()

    def test_valid_question_file_has_no_errors(self):
        payload = {
            "version": "0.1",
            "questions": [
                {
                    "question_id": "scene_0001_q001",
                    "image": "scene_0001.jpg",
                    "question": "Which object is closer, the cup or the laptop?",
                    "type": "closer_farther",
                    "target_objects": ["cup", "laptop"],
                    "answer": "cup",
                    "evaluation": {"metric": "exact_match", "acceptable_answers": ["cup"]},
                }
            ],
        }

        errors = self.module.validate_question_payload(payload)

        self.assertEqual(errors, [])

    def test_missing_required_question_fields_are_reported(self):
        payload = {"version": "0.1", "questions": [{"question_id": "scene_0001_q001"}]}

        errors = self.module.validate_question_payload(payload)

        self.assertIn("questions[0] missing required field: image", errors)
        self.assertIn("questions[0] missing required field: question", errors)
        self.assertIn("questions[0] missing required field: type", errors)
        self.assertIn("questions[0] missing required field: target_objects", errors)
        self.assertIn("questions[0] missing required field: answer", errors)
        self.assertIn("questions[0] missing required field: evaluation", errors)

    def test_load_questions_reads_json_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "questions.json"
            path.write_text(json.dumps({"version": "0.1", "questions": []}), encoding="utf-8")

            payload = self.module.load_questions(path)

        self.assertEqual(payload["version"], "0.1")
        self.assertEqual(payload["questions"], [])


if __name__ == "__main__":
    unittest.main()
