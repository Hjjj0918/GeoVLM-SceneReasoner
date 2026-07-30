import importlib.util
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_depth_module():
    module_path = REPO_ROOT / "scripts" / "09_estimate_depth.py"
    spec = importlib.util.spec_from_file_location("estimate_depth", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EstimateDepthTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.image_dir = self.tmp_path / "data" / "images"
        self.output_dir = self.tmp_path / "outputs" / "depth"
        self.image_dir.mkdir(parents=True)
        self.module = load_depth_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_image(self, name: str, width: int = 64, height: int = 48) -> Path:
        path = self.image_dir / name
        image = np.zeros((height, width, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def touch(self, name: str) -> Path:
        path = self.image_dir / name
        path.write_bytes(b"image")
        return path

    def test_collect_image_files_sorts_images_and_ignores_non_images(self):
        self.touch("scene_0001_view_01.jpg")
        self.touch("notes.txt")
        self.touch("scene_0001_view_00.jpg")

        files = self.module.collect_image_files(self.image_dir)

        self.assertEqual(
            [path.name for path in files],
            ["scene_0001_view_00.jpg", "scene_0001_view_01.jpg"],
        )

    def test_normalize_depth_to_uint8_handles_constant_depth(self):
        depth = np.full((2, 3), 7.5, dtype=np.float32)

        preview = self.module.normalize_depth_to_uint8(depth)

        self.assertEqual(preview.shape, (2, 3))
        self.assertEqual(preview.dtype, np.uint8)
        self.assertEqual(int(preview.max()), 0)

    def test_build_depth_payload_records_paths_and_statistics(self):
        image_path = self.image_dir / "scene_0001_view_00.jpg"
        depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        payload = self.module.build_depth_payload(
            image_path=image_path,
            image_width=2,
            image_height=2,
            model_name="depth-anything/Depth-Anything-V2-Small-hf",
            depth_path=self.output_dir / "scene_0001_view_00.npy",
            visualization_path=self.output_dir / "scene_0001_view_00_preview.jpg",
            depth=depth,
        )

        self.assertEqual(payload["image"], "scene_0001_view_00.jpg")
        self.assertEqual(payload["image_width"], 2)
        self.assertEqual(payload["image_height"], 2)
        self.assertEqual(payload["model"], "depth-anything/Depth-Anything-V2-Small-hf")
        self.assertEqual(payload["depth_path"], str((self.output_dir / "scene_0001_view_00.npy")).replace("\\", "/"))
        self.assertEqual(payload["depth_min"], 1.0)
        self.assertEqual(payload["depth_max"], 4.0)
        self.assertEqual(payload["depth_mean"], 2.5)

    def test_write_depth_outputs_writes_npy_preview_and_json(self):
        image_path = self.write_image("scene_0001_view_00.jpg")
        depth = np.arange(64 * 48, dtype=np.float32).reshape(48, 64)

        json_path = self.module.write_depth_outputs(
            image_path=image_path,
            output_dir=self.output_dir,
            model_name="depth-anything/Depth-Anything-V2-Small-hf",
            depth=depth,
            overwrite=False,
        )

        self.assertEqual(json_path.name, "scene_0001_view_00.json")
        self.assertTrue(json_path.exists())
        self.assertTrue((self.output_dir / "scene_0001_view_00.npy").exists())
        self.assertTrue((self.output_dir / "scene_0001_view_00_preview.jpg").exists())

        saved = np.load(self.output_dir / "scene_0001_view_00.npy")
        self.assertEqual(saved.shape, (48, 64))

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["depth_shape"], [48, 64])

    def test_write_depth_outputs_rejects_existing_output_without_overwrite(self):
        image_path = self.write_image("scene_0001_view_00.jpg")
        self.output_dir.mkdir(parents=True)
        (self.output_dir / "scene_0001_view_00.json").write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.module.write_depth_outputs(
                image_path=image_path,
                output_dir=self.output_dir,
                model_name="depth-anything/Depth-Anything-V2-Small-hf",
                depth=np.zeros((48, 64), dtype=np.float32),
                overwrite=False,
            )

    def test_main_returns_error_when_model_loading_fails(self):
        self.write_image("scene_0001_view_00.jpg")

        def fail_load_depth_estimator(model_name: str, device: str | None = None):
            raise OSError("cannot load depth model")

        original_argv = sys.argv
        original_loader = self.module.load_depth_estimator
        try:
            self.module.load_depth_estimator = fail_load_depth_estimator
            sys.argv = [
                "09_estimate_depth.py",
                "--image-dir",
                str(self.image_dir),
                "--output-dir",
                str(self.output_dir),
            ]

            result = self.module.main()

            self.assertEqual(result, 1)
        finally:
            self.module.load_depth_estimator = original_loader
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
