"""Run YOLO object detection and save per-image GeoVLM detection JSON.

Usage: python scripts/04_detect_objects.py --model yolo11n.pt --device cuda --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import NamedTuple

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
REPO_ROOT = Path(__file__).resolve().parents[1]


class DetectionRecord(NamedTuple):
    label: str
    confidence: float
    bbox_xyxy: list[float]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        (path for path in image_dir.iterdir() if is_image_file(path)),
        key=lambda path: path.name.lower(),
    )


def read_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def to_float(value: object) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def to_float_list(values: object) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def extract_yolo_detections(result: object, min_confidence: float) -> list[DetectionRecord]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = getattr(result, "names", {})
    detections: list[DetectionRecord] = []
    for xyxy, confidence, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
        confidence_value = to_float(confidence)
        if confidence_value < min_confidence:
            continue

        class_index = int(to_float(class_id))
        label = names.get(class_index, str(class_index)) if isinstance(names, dict) else str(class_index)
        detections.append(
            DetectionRecord(
                label=label,
                confidence=round(confidence_value, 6),
                bbox_xyxy=to_float_list(xyxy),
            )
        )
    return detections


def build_detection_payload(
    image_path: Path,
    image_width: int,
    image_height: int,
    detections: list[DetectionRecord],
    model_name: str,
) -> dict:
    return {
        "image": image_path.name,
        "image_width": image_width,
        "image_height": image_height,
        "model": model_name,
        "objects": [
            {
                "object_id": f"obj_{index:03d}",
                "label": detection.label,
                "confidence": detection.confidence,
                "bbox_xyxy": detection.bbox_xyxy,
            }
            for index, detection in enumerate(detections, start=1)
        ],
    }


def output_path_for_image(output_dir: Path, image_path: Path) -> Path:
    return output_dir / f"{image_path.stem}.json"


def write_detection_payload(payload: dict, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_ultralytics_environment(config_dir: Path | None = None) -> None:
    target_dir = config_dir or REPO_ROOT / "outputs" / "ultralytics_config"
    target_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(target_dir.resolve()))


def load_yolo_model(model_name: str):
    prepare_ultralytics_environment()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Install detection dependencies with: "
            "python -m pip install ultralytics"
        ) from error
    return YOLO(model_name)


def run_yolo_on_image(
    model: object,
    image_path: Path,
    confidence: float,
    device: str | None,
    image_size: int | None,
) -> object:
    predict_kwargs = {
        "source": str(image_path),
        "conf": confidence,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device
    if image_size:
        predict_kwargs["imgsz"] = image_size

    results = model.predict(**predict_kwargs)
    if not results:
        raise RuntimeError(f"YOLO returned no result for {image_path}")
    return results[0]


def detect_image(
    model: object,
    image_path: Path,
    model_name: str,
    confidence: float,
    device: str | None,
    image_size: int | None,
) -> dict:
    image_width, image_height = read_image_size(image_path)
    result = run_yolo_on_image(
        model=model,
        image_path=image_path,
        confidence=confidence,
        device=device,
        image_size=image_size,
    )
    detections = extract_yolo_detections(result, min_confidence=confidence)
    return build_detection_payload(
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
        detections=detections,
        model_name=model_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO object detection for GeoVLM scene images.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/detections"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="Optional YOLO device, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--image-size", type=int, default=None, help="Optional YOLO inference image size.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing detection JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="List planned outputs without loading YOLO.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = collect_image_files(args.image_dir)
    if args.limit is not None:
        images = images[: args.limit]

    if not images:
        print(f"No image files found in {args.image_dir}")
        return 1

    print(f"Images: {len(images)}")
    print(f"Output dir: {args.output_dir}")

    if args.dry_run:
        for image_path in images:
            print(f"{image_path.name} -> {output_path_for_image(args.output_dir, image_path)}")
        return 0

    try:
        model = load_yolo_model(args.model)
        for image_path in images:
            output_path = output_path_for_image(args.output_dir, image_path)
            payload = detect_image(
                model=model,
                image_path=image_path,
                model_name=args.model,
                confidence=args.confidence,
                device=args.device,
                image_size=args.image_size,
            )
            write_detection_payload(payload, output_path, overwrite=args.overwrite)
            print(f"Wrote {output_path} ({len(payload['objects'])} objects)")
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
