import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_scaffold_module():
    module_path = REPO_ROOT / "scripts" / "03_scaffold_questions.py"
    spec = importlib.util.spec_from_file_location("scaffold_questions", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScaffoldQuestionsTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.image_dir = self.tmp_path / "data" / "images"
        self.image_dir.mkdir(parents=True)
        self.module = load_scaffold_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def touch(self, name: str) -> Path:
        path = self.image_dir / name
        path.write_bytes(b"image")
        return path

    def template_payload(self) -> dict:
        return {
            "version": "0.1",
            "scene_group": "scene_0001",
            "templates": [
                {
                    "template_id": "q001",
                    "question": "Which object is closer to the camera, the cup or the laptop?",
                    "type": "closer_farther",
                    "target_objects": ["cup", "laptop"],
                    "answer": "cup",
                    "evaluation": {
                        "metric": "exact_match",
                        "acceptable_answers": ["cup"],
                    },
                }
            ],
        }

    def test_collect_image_files_sorts_scene_views(self):
        self.touch("scene_0001_view_01.jpg")
        self.touch(".gitkeep")
        self.touch("rename_manifest.json")
        self.touch("scene_0001_view_00.jpg")

        files = self.module.collect_image_files(self.image_dir)

        self.assertEqual(
            [path.name for path in files],
            ["scene_0001_view_00.jpg", "scene_0001_view_01.jpg"],
        )

    def test_expand_templates_creates_one_question_per_image_per_template(self):
        images = [
            self.touch("scene_0001_view_00.jpg"),
            self.touch("scene_0001_view_01.jpg"),
        ]

        payload = self.module.expand_question_templates(images, self.template_payload())

        self.assertEqual(payload["version"], "0.1")
        self.assertEqual(payload["scene_group"], "scene_0001")
        self.assertEqual(len(payload["questions"]), 2)
        self.assertEqual(payload["questions"][0]["question_id"], "scene_0001_view_00_q001")
        self.assertEqual(payload["questions"][0]["image"], "scene_0001_view_00.jpg")
        self.assertEqual(payload["questions"][0]["scene_group"], "scene_0001")
        self.assertEqual(payload["questions"][0]["camera_variation"], "view_00")
        self.assertEqual(payload["questions"][1]["question_id"], "scene_0001_view_01_q001")

    def test_write_questions_rejects_existing_output_without_force(self):
        output_path = self.tmp_path / "data" / "questions.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.module.write_questions({"questions": []}, output_path, force=False)

    def test_write_questions_allows_force_overwrite(self):
        output_path = self.tmp_path / "data" / "questions.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")

        self.module.write_questions({"version": "0.1", "questions": []}, output_path, force=True)

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], "0.1")


if __name__ == "__main__":
    unittest.main()
