"""Draw YOLO detection boxes for manual review.

Usage: python scripts/05_visualize_detections.py --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def collect_detection_files(detection_dir: Path) -> list[Path]:
    if not detection_dir.exists():
        return []
    return sorted(
        (path for path in detection_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json"),
        key=lambda path: path.name.lower(),
    )


def load_detection_payload(detection_path: Path) -> dict[str, Any]:
    return json.loads(detection_path.read_text(encoding="utf-8"))


def label_color(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    # OpenCV uses BGR. Keep colors bright enough to read on dark or light images.
    return (
        80 + digest[0] % 176,
        80 + digest[1] % 176,
        80 + digest[2] % 176,
    )


def clamp_bbox(bbox_xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(0, min(width - 1, int(round(x2))))
    bottom = max(0, min(height - 1, int(round(y2))))
    return left, top, right, bottom


def draw_detection(image: np.ndarray, detection: dict[str, Any]) -> None:
    height, width = image.shape[:2]
    label = str(detection.get("label", "unknown"))
    object_id = str(detection.get("object_id", "obj"))
    confidence = float(detection.get("confidence", 0.0))
    bbox = detection.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return

    left, top, right, bottom = clamp_bbox(bbox, width=width, height=height)
    color = label_color(label)
    thickness = max(2, round(min(width, height) / 900))
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)

    text = f"{object_id} {label} {confidence:.2f}"
    font_scale = max(0.7, min(width, height) / 1800)
    text_thickness = max(1, round(thickness / 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_thickness,
    )

    label_top = max(0, top - text_height - baseline - 8)
    label_bottom = label_top + text_height + baseline + 8
    label_right = min(width - 1, left + text_width + 8)
    cv2.rectangle(image, (left, label_top), (label_right, label_bottom), color, -1)
    cv2.putText(
        image,
        text,
        (left + 4, label_bottom - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )


def visualize_detection_file(
    detection_path: Path,
    image_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    payload = load_detection_payload(detection_path)
    image_name = payload.get("image")
    if not isinstance(image_name, str):
        raise ValueError(f"Detection file missing image field: {detection_path}")

    image_path = image_dir / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for detection file {detection_path.name}: {image_path}")

    output_path = output_dir / image_name
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    for detection in payload.get("objects", []):
        if isinstance(detection, dict):
            draw_detection(image, detection)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"OpenCV could not write image: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GeoVLM detection JSON outputs.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--detection-dir", type=Path, default=Path("outputs/detections"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations/detections"))
    parser.add_argument("--limit", type=int, default=None, help="Visualize only the first N detection files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing visualization images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detection_files = collect_detection_files(args.detection_dir)
    if args.limit is not None:
        detection_files = detection_files[: args.limit]

    if not detection_files:
        print(f"No detection JSON files found in {args.detection_dir}")
        return 1

    try:
        for detection_path in detection_files:
            output_path = visualize_detection_file(
                detection_path=detection_path,
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
            )
            print(f"Wrote {output_path}")
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print(f"Visualized {len(detection_files)} detection files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
