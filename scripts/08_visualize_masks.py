"""Draw SAM2 mask overlays for manual review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def collect_segment_files(mask_dir: Path) -> list[Path]:
    if not mask_dir.exists():
        return []
    return sorted(mask_dir.rglob("segments.json"), key=lambda path: str(path.relative_to(mask_dir)).lower())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def label_color(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (
        80 + digest[0] % 176,
        80 + digest[1] % 176,
        80 + digest[2] % 176,
    )


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    return image


def resolve_mask_path(segments_path: Path, mask_path_value: Any) -> Path:
    if not isinstance(mask_path_value, str) or not mask_path_value.strip():
        raise ValueError(f"Invalid mask_path in {segments_path}")

    candidate = Path(mask_path_value)
    if candidate.is_absolute():
        return candidate

    repo_candidate = (REPO_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    local_candidate = (segments_path.parent / candidate.name).resolve()
    if local_candidate.exists():
        return local_candidate

    return repo_candidate


def read_mask(mask_path: Path, image_width: int, image_height: int) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"OpenCV could not read mask: {mask_path}")
    if mask.shape != (image_height, image_width):
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, 0, 0
    left = int(xs.min())
    right = int(xs.max())
    top = int(ys.min())
    bottom = int(ys.max())
    return left, top, right, bottom


def draw_label(
    image: np.ndarray,
    text: str,
    left: int,
    top: int,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    thickness = max(1, round(min(width, height) / 900))
    font_scale = max(0.7, min(width, height) / 1800)
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    label_top = max(0, top - text_height - baseline - 8)
    label_bottom = min(height - 1, label_top + text_height + baseline + 8)
    label_right = min(width - 1, left + text_width + 8)
    cv2.rectangle(image, (left, label_top), (label_right, label_bottom), color, -1)
    cv2.putText(
        image,
        text,
        (left + 4, label_bottom - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy()
    tint = np.full_like(image, color)
    blended = cv2.addWeighted(image, 1.0 - alpha, tint, alpha, 0)
    overlay[mask] = blended[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, max(1, round(min(image.shape[:2]) / 1000)))
    return overlay


def visualize_mask_file(
    segments_path: Path,
    image_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    payload = load_json(segments_path)
    image_name = payload.get("image")
    if not isinstance(image_name, str):
        raise ValueError(f"Segments file missing image field: {segments_path}")

    image_path = image_dir / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for segments file {segments_path.name}: {image_path}")

    output_path = output_dir / image_name
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    image = load_image(image_path)
    image_height, image_width = image.shape[:2]

    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        objects = []

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        mask_path = resolve_mask_path(segments_path, obj.get("mask_path"))
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {segments_path.name}: {mask_path}")

        mask = read_mask(mask_path, image_width=image_width, image_height=image_height)
        color = label_color(str(obj.get("label", "unknown")))
        image = overlay_mask(image, mask, color=color)

        left, top, _, _ = mask_bounds(mask)
        object_id = str(obj.get("object_id", "obj"))
        label = str(obj.get("label", "unknown"))
        mask_area_px = obj.get("mask_area_px", 0)
        draw_label(image, f"{object_id} {label} {mask_area_px}", left, top, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"OpenCV could not write image: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GeoVLM SAM2 mask outputs.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--mask-dir", type=Path, default=Path("outputs/masks"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations/masks"))
    parser.add_argument("--limit", type=int, default=None, help="Visualize only the first N mask files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing visualization images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    segment_files = collect_segment_files(args.mask_dir)
    if args.limit is not None:
        segment_files = segment_files[: args.limit]

    if not segment_files:
        print(f"No segments.json files found in {args.mask_dir}")
        return 1

    try:
        for segments_path in segment_files:
            output_path = visualize_mask_file(
                segments_path=segments_path,
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
            )
            print(f"Wrote {output_path}")
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print(f"Visualized {len(segment_files)} mask files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
