"""Extract object-level geometry from masks and relative depth maps.

Usage: python scripts/10_extract_geometry.py --overwrite
"""

from __future__ import annotations

import argparse
import itertools
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


def normalized_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


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


def resolve_depth_path(depth_dir: Path, image_name: str) -> Path:
    return depth_dir / f"{Path(image_name).stem}.npy"


def read_mask(mask_path: Path, image_width: int, image_height: int) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"OpenCV could not read mask: {mask_path}")
    if mask.shape != (image_height, image_width):
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def read_depth(depth_path: Path, image_width: int, image_height: int) -> np.ndarray:
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape != (image_height, image_width):
        depth = cv2.resize(depth, (image_width, image_height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return depth


def finite_depth_range(depth: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(depth, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0
    depth_min = float(finite.min())
    depth_max = float(finite.max())
    return depth_min, depth_max, depth_max - depth_min


def bbox_center(bbox_xyxy: Any) -> list[float] | None:
    if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
        return None
    left, top, right, bottom = [float(value) for value in bbox_xyxy]
    return [round_float((left + right) / 2.0), round_float((top + bottom) / 2.0)]


def compute_mask_geometry(mask: np.ndarray, image_width: int, image_height: int) -> dict[str, Any]:
    ys, xs = np.where(mask)
    area = int(xs.size)
    if area == 0:
        return {
            "mask_area_px": 0,
            "mask_area_fraction": 0.0,
            "mask_bounds_xyxy": None,
            "mask_centroid": None,
        }

    left = int(xs.min())
    right = int(xs.max())
    top = int(ys.min())
    bottom = int(ys.max())
    return {
        "mask_area_px": area,
        "mask_area_fraction": round_float(area / float(image_width * image_height)),
        "mask_bounds_xyxy": [left, top, right, bottom],
        "mask_centroid": [round_float(float(xs.mean())), round_float(float(ys.mean()))],
    }


def compute_depth_stats(depth: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(depth, dtype=np.float32)[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "relative_depth_min": None,
            "relative_depth_max": None,
            "relative_depth_mean": None,
            "relative_depth_median": None,
            "relative_depth_p10": None,
            "relative_depth_p90": None,
        }
    return {
        "relative_depth_min": round_float(float(values.min())),
        "relative_depth_max": round_float(float(values.max())),
        "relative_depth_mean": round_float(float(values.mean())),
        "relative_depth_median": round_float(float(np.median(values))),
        "relative_depth_p10": round_float(float(np.percentile(values, 10))),
        "relative_depth_p90": round_float(float(np.percentile(values, 90))),
    }


def position_tag(value: float | None, extent: int, low_label: str, mid_label: str, high_label: str) -> str:
    if value is None:
        return "unknown"
    ratio = value / float(extent)
    if ratio < 1.0 / 3.0:
        return low_label
    if ratio > 2.0 / 3.0:
        return high_label
    return mid_label


def depth_percentile(
    relative_depth_median: float | None,
    depth_min: float,
    depth_range: float,
) -> float | None:
    if relative_depth_median is None or depth_range <= 0:
        return None
    return round_float((relative_depth_median - depth_min) / depth_range)


def depth_order_hint(percentile: float | None, higher_depth_is_closer: bool) -> str:
    if percentile is None:
        return "unknown"
    near_value = percentile if higher_depth_is_closer else 1.0 - percentile
    if near_value >= 2.0 / 3.0:
        return "near"
    if near_value <= 1.0 / 3.0:
        return "far"
    return "middle"


def build_object_geometry(
    segment: dict[str, Any],
    mask: np.ndarray,
    depth: np.ndarray,
    image_width: int,
    image_height: int,
    depth_min: float,
    depth_range: float,
    higher_depth_is_closer: bool,
) -> dict[str, Any]:
    mask_geometry = compute_mask_geometry(mask, image_width=image_width, image_height=image_height)
    depth_stats = compute_depth_stats(depth, mask)
    centroid = mask_geometry["mask_centroid"]
    center = bbox_center(segment.get("bbox_xyxy"))
    if center is None:
        center = centroid

    percentile = depth_percentile(
        depth_stats["relative_depth_median"],
        depth_min=depth_min,
        depth_range=depth_range,
    )
    x_value = centroid[0] if centroid else None
    y_value = centroid[1] if centroid else None

    record = {
        "object_id": segment.get("object_id"),
        "label": segment.get("label"),
        "confidence": segment.get("confidence"),
        "bbox_xyxy": segment.get("bbox_xyxy"),
        "bbox_center": center,
        **mask_geometry,
        **depth_stats,
        "relative_depth_percentile": percentile,
        "horizontal_position": position_tag(x_value, image_width, "left", "center", "right"),
        "vertical_position": position_tag(y_value, image_height, "top", "middle", "bottom"),
        "depth_order_hint": depth_order_hint(percentile, higher_depth_is_closer),
    }
    if "raw_label" in segment:
        record["raw_label"] = segment["raw_label"]
    if "normalization" in segment:
        record["normalization"] = segment["normalization"]
    return record


def compare_axis(a_value: float, b_value: float, threshold: float, low_relation: str, high_relation: str) -> str:
    delta = a_value - b_value
    if abs(delta) <= threshold:
        return "aligned"
    return high_relation if delta > 0 else low_relation


def compare_depth(
    a_depth: float | None,
    b_depth: float | None,
    threshold: float,
    higher_depth_is_closer: bool,
) -> str:
    if a_depth is None or b_depth is None:
        return "unknown"
    delta = a_depth - b_depth
    if abs(delta) <= threshold:
        return "similar_depth"
    if higher_depth_is_closer:
        return "closer_than" if delta > 0 else "farther_than"
    return "closer_than" if delta < 0 else "farther_than"


def build_pairwise_relations(
    objects: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    depth_range: float,
    higher_depth_is_closer: bool,
) -> list[dict[str, Any]]:
    relations = []
    horizontal_threshold = max(1.0, image_width * 0.03)
    vertical_threshold = max(1.0, image_height * 0.03)
    depth_threshold = max(1e-6, depth_range * 0.02)

    for object_a, object_b in itertools.combinations(objects, 2):
        centroid_a = object_a.get("mask_centroid")
        centroid_b = object_b.get("mask_centroid")
        if not centroid_a or not centroid_b:
            continue
        relations.append(
            {
                "object_a": object_a.get("object_id"),
                "label_a": object_a.get("label"),
                "object_b": object_b.get("object_id"),
                "label_b": object_b.get("label"),
                "horizontal_relation": compare_axis(
                    float(centroid_a[0]),
                    float(centroid_b[0]),
                    horizontal_threshold,
                    "left_of",
                    "right_of",
                ),
                "vertical_relation": compare_axis(
                    float(centroid_a[1]),
                    float(centroid_b[1]),
                    vertical_threshold,
                    "above",
                    "below",
                ),
                "depth_relation": compare_depth(
                    object_a.get("relative_depth_median"),
                    object_b.get("relative_depth_median"),
                    depth_threshold,
                    higher_depth_is_closer,
                ),
            }
        )
    return relations


def depth_order_assumption(higher_depth_is_closer: bool) -> str:
    if higher_depth_is_closer:
        return "higher_relative_depth_is_closer"
    return "lower_relative_depth_is_closer"


def extract_geometry_file(
    segments_path: Path,
    depth_dir: Path,
    output_dir: Path,
    overwrite: bool,
    higher_depth_is_closer: bool,
) -> Path:
    segment_payload = load_json(segments_path)
    image_name = segment_payload.get("image")
    if not isinstance(image_name, str):
        raise ValueError(f"Segments file missing image field: {segments_path}")

    image_width = int(segment_payload.get("image_width", 0))
    image_height = int(segment_payload.get("image_height", 0))
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Segments file has invalid image dimensions: {segments_path}")

    output_path = output_dir / f"{Path(image_name).stem}.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Geometry output already exists: {output_path}")

    depth_path = resolve_depth_path(depth_dir, image_name)
    depth = read_depth(depth_path, image_width=image_width, image_height=image_height)
    depth_min, depth_max, depth_range = finite_depth_range(depth)

    objects = []
    for segment in segment_payload.get("objects", []):
        if not isinstance(segment, dict):
            continue
        mask_path = resolve_mask_path(segments_path, segment.get("mask_path"))
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {segments_path.name}: {mask_path}")
        mask = read_mask(mask_path, image_width=image_width, image_height=image_height)
        objects.append(
            build_object_geometry(
                segment=segment,
                mask=mask,
                depth=depth,
                image_width=image_width,
                image_height=image_height,
                depth_min=depth_min,
                depth_range=depth_range,
                higher_depth_is_closer=higher_depth_is_closer,
            )
        )

    payload = {
        "image": image_name,
        "image_width": image_width,
        "image_height": image_height,
        "source_segments": normalized_path(segments_path),
        "source_depth": normalized_path(depth_path),
        "depth_order_assumption": depth_order_assumption(higher_depth_is_closer),
        "image_depth_min": round_float(depth_min),
        "image_depth_max": round_float(depth_max),
        "image_depth_range": round_float(depth_range),
        "objects": objects,
        "pairwise_relations": build_pairwise_relations(
            objects,
            image_width=image_width,
            image_height=image_height,
            depth_range=depth_range,
            higher_depth_is_closer=higher_depth_is_closer,
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GeoVLM object-level geometry from masks and depth maps.")
    parser.add_argument("--mask-dir", type=Path, default=Path("outputs/masks"))
    parser.add_argument("--depth-dir", type=Path, default=Path("outputs/depth"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/geometry"))
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N segment files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing geometry JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="List planned outputs without reading masks or depth.")
    parser.add_argument(
        "--lower-depth-is-closer",
        action="store_true",
        help="Invert depth ordering if manual review shows lower relative depth means closer.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    segment_files = collect_segment_files(args.mask_dir)
    if args.limit is not None:
        segment_files = segment_files[: args.limit]

    if not segment_files:
        print(f"No segments.json files found in {args.mask_dir}")
        return 1

    print(f"Segment files: {len(segment_files)}")
    print(f"Depth dir: {args.depth_dir}")
    print(f"Output dir: {args.output_dir}")

    if args.dry_run:
        for segments_path in segment_files:
            payload = load_json(segments_path)
            image_name = payload.get("image", f"{segments_path.parent.name}.jpg")
            print(f"{segments_path} -> {args.output_dir / (Path(str(image_name)).stem + '.json')}")
        return 0

    try:
        for segments_path in segment_files:
            output_path = extract_geometry_file(
                segments_path=segments_path,
                depth_dir=args.depth_dir,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                higher_depth_is_closer=not args.lower_depth_is_closer,
            )
            payload = load_json(output_path)
            print(f"Wrote {output_path} ({len(payload['objects'])} objects)")
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
