"""Check CalibVLM-lite dataset folders before running the pipeline."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import NamedTuple

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ImageRecord(NamedTuple):
    path: Path
    width: int
    height: int
    has_recommended_name: bool


class DatasetSummary(NamedTuple):
    total_images: int
    unique_dimensions: list[tuple[int, int]]
    has_inconsistent_dimensions: bool


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def has_recommended_name(path: Path, expected_prefix: str) -> bool:
    pattern = rf"^{re.escape(expected_prefix)}_\d{{4}}\.(jpg|jpeg|png|bmp)$"
    return re.match(pattern, path.name, flags=re.IGNORECASE) is not None


def read_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    height, width = image.shape[:2]
    return width, height


def collect_image_records(directory: Path, expected_prefix: str) -> list[ImageRecord]:
    if not directory.exists():
        return []

    records: list[ImageRecord] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if not is_image_file(path):
            continue
        width, height = read_image_size(path)
        records.append(
            ImageRecord(
                path=path,
                width=width,
                height=height,
                has_recommended_name=has_recommended_name(path, expected_prefix),
            )
        )
    return records


def summarize_dimensions(records: list[ImageRecord]) -> DatasetSummary:
    unique_dimensions = sorted({(record.width, record.height) for record in records})
    return DatasetSummary(
        total_images=len(records),
        unique_dimensions=unique_dimensions,
        has_inconsistent_dimensions=len(unique_dimensions) > 1,
    )


def print_section(title: str, records: list[ImageRecord]) -> None:
    summary = summarize_dimensions(records)
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Images: {summary.total_images}")

    if summary.unique_dimensions:
        dims = ", ".join(f"{width}x{height}" for width, height in summary.unique_dimensions)
        print(f"Dimensions: {dims}")
    else:
        print("Dimensions: none")

    if summary.has_inconsistent_dimensions:
        print("Warning: image dimensions are inconsistent.")

    bad_names = [record.path.name for record in records if not record.has_recommended_name]
    if bad_names:
        print("Warning: files outside recommended naming pattern:")
        for name in bad_names:
            print(f"  - {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CalibVLM-lite dataset folders.")
    parser.add_argument("--scene-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("data/calibration"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_records = collect_image_records(args.scene_dir, expected_prefix="scene")
    calibration_records = collect_image_records(args.calibration_dir, expected_prefix="calib")

    print("CalibVLM-lite dataset check")
    print_section("Scene images", scene_records)
    print_section("Calibration images", calibration_records)

    if not scene_records:
        print("\nNext: add desktop scene images to data/images/ as scene_0001.jpg, scene_0002.jpg, ...")
    if not calibration_records:
        print("Next: add chessboard calibration images to data/calibration/ as calib_0001.jpg, calib_0002.jpg, ...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
