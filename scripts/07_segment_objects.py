"""Run SAM2 segmentation from normalized detection boxes.

Usage: python scripts/07_segment_objects.py --model sam2_t.pt --device cuda --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def collect_detection_files(detection_dir: Path) -> list[Path]:
    if not detection_dir.exists():
        return []
    return sorted(
        (path for path in detection_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json"),
        key=lambda path: path.name.lower(),
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def prepare_box_prompt(detection: dict[str, Any]) -> list[float]:
    bbox = detection.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Detection has invalid bbox_xyxy: {detection}")
    return [float(value) for value in bbox]


def to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def mask_to_uint8(mask: object, width: int, height: int) -> np.ndarray:
    mask_array = to_numpy(mask)
    mask_array = np.squeeze(mask_array)
    if mask_array.shape != (height, width):
        mask_array = cv2.resize(mask_array.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask_array > 0.5).astype(np.uint8) * 255


def extract_first_mask(result: object) -> object:
    masks = getattr(result, "masks", None)
    if masks is None:
        raise RuntimeError("SAM2 returned no masks.")
    mask_data = getattr(masks, "data", None)
    if mask_data is None:
        raise RuntimeError("SAM2 returned masks without data.")
    if len(mask_data) == 0:
        raise RuntimeError("SAM2 returned an empty mask list.")
    return mask_data[0]


def run_sam2_for_box(model: object, image_path: Path, bbox_xyxy: list[float], device: str | None) -> object:
    predict_kwargs = {
        "source": str(image_path),
        "bboxes": [bbox_xyxy],
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device
    results = model.predict(**predict_kwargs)
    if not results:
        raise RuntimeError(f"SAM2 returned no result for {image_path}")
    return extract_first_mask(results[0])


def mask_path_for_object(segment_dir: Path, object_id: str) -> Path:
    return segment_dir / f"{object_id}.png"


def build_segment_record(detection: dict[str, Any], mask_path: Path, mask_area_px: int) -> dict[str, Any]:
    record = {
        "object_id": detection["object_id"],
        "label": detection["label"],
        "confidence": detection.get("confidence"),
        "bbox_xyxy": detection["bbox_xyxy"],
        "mask_path": str(mask_path).replace("\\", "/"),
        "mask_area_px": mask_area_px,
    }
    if "raw_label" in detection:
        record["raw_label"] = detection["raw_label"]
    if "normalization" in detection:
        record["normalization"] = detection["normalization"]
    return record


def write_mask(mask: np.ndarray, mask_path: Path, overwrite: bool) -> None:
    if mask_path.exists() and not overwrite:
        raise FileExistsError(f"Mask already exists: {mask_path}")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_path), mask):
        raise ValueError(f"OpenCV could not write mask: {mask_path}")


def write_segments_json(
    segment_dir: Path,
    image_name: str,
    image_width: int,
    image_height: int,
    source_detection: str,
    records: list[dict[str, Any]],
    model_name: str,
    overwrite: bool,
) -> Path:
    output_path = segment_dir / "segments.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Segments file already exists: {output_path}")
    segment_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": image_name,
        "image_width": image_width,
        "image_height": image_height,
        "source_detection": source_detection.replace("\\", "/"),
        "segmentation_model": model_name,
        "objects": records,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def segment_detection_file(
    model: object,
    detection_path: Path,
    image_dir: Path,
    output_dir: Path,
    model_name: str,
    overwrite: bool,
    device: str | None = None,
) -> Path:
    detection_payload = load_json(detection_path)
    image_name = detection_payload.get("image")
    if not isinstance(image_name, str):
        raise ValueError(f"Detection file missing image field: {detection_path}")

    image_path = image_dir / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for detection file {detection_path.name}: {image_path}")

    image_width, image_height = read_image_size(image_path)
    segment_dir = output_dir / Path(image_name).stem
    records: list[dict[str, Any]] = []

    objects = detection_payload.get("objects", [])
    if not isinstance(objects, list):
        objects = []

    for detection in objects:
        if not isinstance(detection, dict):
            continue
        bbox_xyxy = prepare_box_prompt(detection)
        object_id = str(detection["object_id"])
        mask_path = mask_path_for_object(segment_dir, object_id)
        raw_mask = run_sam2_for_box(
            model=model,
            image_path=image_path,
            bbox_xyxy=bbox_xyxy,
            device=device,
        )
        mask = mask_to_uint8(raw_mask, width=image_width, height=image_height)
        write_mask(mask, mask_path, overwrite=overwrite)
        records.append(
            build_segment_record(
                detection=detection,
                mask_path=mask_path,
                mask_area_px=int(np.count_nonzero(mask)),
            )
        )

    return write_segments_json(
        segment_dir=segment_dir,
        image_name=image_name,
        image_width=image_width,
        image_height=image_height,
        source_detection=str(detection_path),
        records=records,
        model_name=model_name,
        overwrite=overwrite,
    )


def prepare_ultralytics_environment(config_dir: Path | None = None) -> None:
    target_dir = config_dir or REPO_ROOT / "outputs" / "ultralytics_config"
    target_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(target_dir.resolve()))


def load_sam_model(model_name: str):
    prepare_ultralytics_environment()
    try:
        from ultralytics import SAM
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Install segmentation dependencies with: "
            "python -m pip install ultralytics"
        ) from error
    return SAM(model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM2 segmentation from normalized GeoVLM detections.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--detection-dir", type=Path, default=Path("outputs/detections_normalized"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/masks"))
    parser.add_argument("--model", default="sam2_t.pt")
    parser.add_argument("--device", default=None, help="Optional device, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--limit", type=int, default=None, help="Segment only the first N detection files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks and segments files.")
    parser.add_argument("--dry-run", action="store_true", help="List planned segment outputs without loading SAM2.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detection_files = collect_detection_files(args.detection_dir)
    if args.limit is not None:
        detection_files = detection_files[: args.limit]

    if not detection_files:
        print(f"No detection JSON files found in {args.detection_dir}")
        return 1

    print(f"Detection files: {len(detection_files)}")
    print(f"Output dir: {args.output_dir}")

    if args.dry_run:
        for detection_path in detection_files:
            payload = load_json(detection_path)
            image_name = payload.get("image", detection_path.with_suffix(".jpg").name)
            print(f"{detection_path.name} -> {args.output_dir / Path(str(image_name)).stem / 'segments.json'}")
        return 0

    try:
        model = load_sam_model(args.model)
        for detection_path in detection_files:
            output_path = segment_detection_file(
                model=model,
                detection_path=detection_path,
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                model_name=args.model,
                overwrite=args.overwrite,
                device=args.device,
            )
            payload = load_json(output_path)
            print(f"Wrote {output_path} ({len(payload['objects'])} masks)")
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
