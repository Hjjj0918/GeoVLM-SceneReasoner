import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_calibration_module():
    module_path = REPO_ROOT / "scripts" / "01_calibrate_camera.py"
    spec = importlib.util.spec_from_file_location("calibrate_camera", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CameraCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.module = load_calibration_module()

    def test_generate_object_points_uses_inner_corner_grid_and_square_size(self):
        points = self.module.generate_object_points(rows=9, cols=6, square_size_m=0.025)

        self.assertEqual(points.shape, (54, 3))
        self.assertEqual(points.dtype, np.float32)
        np.testing.assert_allclose(points[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(points[1], [0.025, 0.0, 0.0])
        np.testing.assert_allclose(points[9], [0.0, 0.025, 0.0])

    def test_build_intrinsics_payload_exposes_geometry_fields(self):
        camera_matrix = np.array(
            [
                [1000.0, 0.0, 320.0],
                [0.0, 990.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.array([[0.1, -0.2, 0.0, 0.0, 0.01]], dtype=np.float64)

        payload = self.module.build_intrinsics_payload(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
            image_size=(640, 480),
            reprojection_error=0.42,
            calibration_images_used=12,
            rows=9,
            cols=6,
            square_size_m=0.025,
        )

        self.assertEqual(payload["camera_model"], "pinhole")
        self.assertEqual(payload["image_width"], 640)
        self.assertEqual(payload["image_height"], 480)
        self.assertEqual(payload["fx"], 1000.0)
        self.assertEqual(payload["fy"], 990.0)
        self.assertEqual(payload["cx"], 320.0)
        self.assertEqual(payload["cy"], 240.0)
        self.assertEqual(payload["distortion_coefficients"], [0.1, -0.2, 0.0, 0.0, 0.01])
        self.assertEqual(payload["calibration_images_used"], 12)
        self.assertEqual(payload["chessboard"]["inner_corners_per_row"], 9)
        self.assertEqual(payload["chessboard"]["inner_corners_per_col"], 6)

    def test_require_minimum_valid_images_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Need at least 3 valid calibration images"):
            self.module.require_minimum_valid_images(valid_count=2, min_valid_images=3)


if __name__ == "__main__":
    unittest.main()
