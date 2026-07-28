"""Calibrate a pinhole camera from chessboard images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_image_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda item: item.name.lower(),
    )


def generate_object_points(rows: int, cols: int, square_size_m: float) -> np.ndarray:
    object_points = np.zeros((rows * cols, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    object_points *= square_size_m
    return object_points


def require_minimum_valid_images(valid_count: int, min_valid_images: int) -> None:
    if valid_count < min_valid_images:
        raise ValueError(
            f"Need at least {min_valid_images} valid calibration images, but only found {valid_count}. "
            "Capture more chessboard images or check --rows/--cols."
        )


def detect_corners(image_path: Path, rows: int, cols: int) -> tuple[bool, np.ndarray | None, tuple[int, int] | None]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
    if not found:
        return False, None, (gray.shape[1], gray.shape[0])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined, (gray.shape[1], gray.shape[0])


def compute_mean_reprojection_error(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> float:
    total_error = 0.0
    total_points = 0
    for obj_points, img_points, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, distortion_coefficients)
        error = cv2.norm(img_points, projected, cv2.NORM_L2)
        total_error += error * error
        total_points += len(obj_points)
    return float(np.sqrt(total_error / total_points)) if total_points else 0.0


def build_intrinsics_payload(
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    image_size: tuple[int, int],
    reprojection_error: float,
    calibration_images_used: int,
    rows: int,
    cols: int,
    square_size_m: float,
) -> dict:
    width, height = image_size
    distortion = distortion_coefficients.reshape(-1).astype(float).tolist()
    return {
        "camera_model": "pinhole",
        "image_width": int(width),
        "image_height": int(height),
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "camera_matrix": camera_matrix.astype(float).tolist(),
        "distortion_coefficients": distortion,
        "reprojection_error": float(reprojection_error),
        "calibration_images_used": int(calibration_images_used),
        "chessboard": {
            "inner_corners_per_row": int(rows),
            "inner_corners_per_col": int(cols),
            "square_size_m": float(square_size_m),
        },
    }


def calibrate_from_images(
    image_paths: list[Path],
    rows: int,
    cols: int,
    square_size_m: float,
    min_valid_images: int,
) -> dict:
    template_object_points = generate_object_points(rows, cols, square_size_m)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for path in image_paths:
        found, corners, current_size = detect_corners(path, rows, cols)
        if current_size is not None and image_size is None:
            image_size = current_size
        if current_size is not None and image_size is not None and current_size != image_size:
            raise ValueError(f"Calibration image size mismatch: {path} is {current_size}, expected {image_size}.")
        if found and corners is not None:
            object_points.append(template_object_points.copy())
            image_points.append(corners)
            print(f"Detected chessboard corners: {path.name}")
        else:
            print(f"Skipped image without detected chessboard: {path.name}")

    require_minimum_valid_images(len(object_points), min_valid_images)
    if image_size is None:
        raise ValueError("No readable calibration images found.")

    _, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    reprojection_error = compute_mean_reprojection_error(
        object_points,
        image_points,
        rvecs,
        tvecs,
        camera_matrix,
        distortion,
    )
    return build_intrinsics_payload(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        image_size=image_size,
        reprojection_error=reprojection_error,
        calibration_images_used=len(object_points),
        rows=rows,
        cols=cols,
        square_size_m=square_size_m,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from chessboard images.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/calibration"))
    parser.add_argument("--output", type=Path, default=Path("configs/camera_intrinsics.json"))
    parser.add_argument("--rows", type=int, default=9, help="Inner corners per chessboard row.")
    parser.add_argument("--cols", type=int, default=6, help="Inner corners per chessboard column.")
    parser.add_argument("--square-size-m", type=float, default=0.025, help="Chessboard square size in meters.")
    parser.add_argument("--min-valid-images", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths = find_image_paths(args.input_dir)
    if not image_paths:
        print(f"No calibration images found in {args.input_dir}.")
        print("Add chessboard images named calib_0001.jpg, calib_0002.jpg, ... and rerun this script.")
        return 1

    try:
        payload = calibrate_from_images(
            image_paths=image_paths,
            rows=args.rows,
            cols=args.cols,
            square_size_m=args.square_size_m,
            min_valid_images=args.min_valid_images,
        )
    except ValueError as exc:
        print(f"Calibration failed: {exc}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote camera intrinsics: {args.output}")
    print(f"Valid calibration images: {payload['calibration_images_used']}")
    print(f"Mean reprojection error: {payload['reprojection_error']:.4f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
