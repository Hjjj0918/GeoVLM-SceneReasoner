"""Test stable sequential renaming for images under data/images.

Usage: python -m pytest tests/test_rename_images.py -q
"""

import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_rename_module():
    module_path = REPO_ROOT / "scripts" / "02_rename_images.py"
    spec = importlib.util.spec_from_file_location("rename_images", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RenameImagesTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.image_dir = self.tmp_path / "data" / "images"
        self.image_dir.mkdir(parents=True)
        self.module = load_rename_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def touch(self, name: str) -> Path:
        path = self.image_dir / name
        path.write_bytes(b"image")
        return path

    def test_collect_image_files_sorts_images_and_ignores_non_images(self):
        self.touch("微信图片_20260729115915_140_271.jpg")
        self.touch(".gitkeep")
        self.touch("notes.txt")
        self.touch("微信图片_20260729115807_138_271.jpg")

        files = self.module.collect_image_files(self.image_dir)

        self.assertEqual(
            [path.name for path in files],
            [
                "微信图片_20260729115807_138_271.jpg",
                "微信图片_20260729115915_140_271.jpg",
            ],
        )

    def test_build_rename_plan_uses_scene_view_names(self):
        files = [
            self.touch("微信图片_20260729115807_138_271.jpg"),
            self.touch("微信图片_20260729115811_139_271.png"),
        ]

        plan = self.module.build_rename_plan(files, scene_id="scene_0001")

        self.assertEqual(plan[0].destination.name, "scene_0001_view_00.jpg")
        self.assertEqual(plan[1].destination.name, "scene_0001_view_01.png")
        self.assertEqual(plan[0].source.name, "微信图片_20260729115807_138_271.jpg")

    def test_validate_plan_rejects_destination_collision(self):
        source = self.touch("微信图片_20260729115807_138_271.jpg")
        self.touch("scene_0001_view_00.jpg")
        plan = self.module.build_rename_plan([source], scene_id="scene_0001")

        errors = self.module.validate_rename_plan(plan)

        self.assertIn("destination already exists: scene_0001_view_00.jpg", errors)

    def test_apply_rename_plan_writes_manifest(self):
        self.touch("微信图片_20260729115807_138_271.jpg")
        self.touch("微信图片_20260729115811_139_271.jpg")
        files = self.module.collect_image_files(self.image_dir)
        plan = self.module.build_rename_plan(files, scene_id="scene_0001")
        manifest_path = self.image_dir / "rename_manifest.json"

        self.module.apply_rename_plan(plan, manifest_path)

        self.assertTrue((self.image_dir / "scene_0001_view_00.jpg").exists())
        self.assertTrue((self.image_dir / "scene_0001_view_01.jpg").exists())
        self.assertFalse((self.image_dir / "微信图片_20260729115807_138_271.jpg").exists())

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["scene_id"], "scene_0001")
        self.assertEqual(manifest["renamed_count"], 2)
        self.assertEqual(
            manifest["files"][0],
            {
                "source": "微信图片_20260729115807_138_271.jpg",
                "destination": "scene_0001_view_00.jpg",
            },
        )


if __name__ == "__main__":
    unittest.main()
