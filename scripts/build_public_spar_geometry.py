"""Build answer-free oracle RGB-D evidence records for selected SPAR rows."""

from __future__ import annotations

import math
import json
import re
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from scripts.public_spar_common import json_safe, to_list, write_json
from scripts.inspect_public_spar import iter_dataset_rows


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    try:
        return array.astype(np.float64)
    except (TypeError, ValueError):
        return None


def _matrix_status(value: Any, expected_shape: tuple[int, int]) -> dict[str, Any]:
    array = _array(value)
    if array is not None and array.ndim == 1:
        expected_size = expected_shape[0] * expected_shape[1]
        if array.size == expected_size:
            array = array.reshape(expected_shape)
        elif array.size == 16:
            # SPAR stores calibrated matrices as flattened homogeneous 4x4
            # arrays for both camera intrinsics and poses.
            array = array.reshape((4, 4))
    if array is None or array.ndim != 2 or array.shape != expected_shape:
        shape = list(array.shape) if array is not None else None
        return {"valid": False, "shape": shape}
    finite = bool(np.isfinite(array).all())
    return {
        "valid": finite,
        "shape": list(array.shape),
        "values": json_safe(array.tolist()) if finite else None,
    }


def validate_intrinsics(value: Any) -> dict[str, Any]:
    array = _array(value)
    if array is not None and array.ndim == 1 and array.size == 16:
        array = array.reshape((4, 4))
    if array is not None and array.shape == (4, 4):
        finite = bool(np.isfinite(array).all())
        return {
            "valid": finite,
            "shape": [4, 4],
            "values": json_safe(array.tolist()) if finite else None,
        }
    return _matrix_status(value, (3, 3))


def validate_pose(value: Any) -> dict[str, Any]:
    return _matrix_status(value, (4, 4))


def _marker_mask(image: np.ndarray, color: str) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return np.zeros(rgb.shape[:2], dtype=bool) if rgb.ndim >= 2 else np.zeros((0, 0), dtype=bool)
    rgb = rgb[..., :3].astype(np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    channels = {"red": red, "green": green, "blue": blue}
    channel = channels[color]
    other_channels = [value for name, value in channels.items() if name != color]

    # SPAR overlays are highly saturated, while scene colors such as wood,
    # carpet, and skin can have a large but much weaker channel difference.
    # Keep the lower green cutoff because the JPEG overlay is often around
    # RGB [0, 128, 0], not the full [0, 255, 0].
    minimum = {"red": 220, "green": 100, "blue": 180}[color]
    dominance = {"red": 150, "green": 100, "blue": 150}[color]
    mask = (channel >= minimum) & (channel - np.maximum(other_channels[0], other_channels[1]) >= dominance)

    # Restrict each color to its largest connected annotation component. This
    # prevents a scattered collection of similarly colored scene pixels from
    # shifting the marker centroid.
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if labels_count <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def _sample_depth(depth: np.ndarray, x: float, y: float, radius: int = 1) -> tuple[float | None, int]:
    height, width = depth.shape[:2]
    center_x = int(round(x))
    center_y = int(round(y))
    for current_radius in (radius, max(radius, 3)):
        left, right = max(0, center_x - current_radius), min(width, center_x + current_radius + 1)
        top, bottom = max(0, center_y - current_radius), min(height, center_y + current_radius + 1)
        values = depth[top:bottom, left:right]
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            return round(float(np.median(values)), 6), int(values.size)
    return None, 0


def _camera_xyz(x: float, y: float, depth: float, intrinsics: np.ndarray | None) -> list[float] | None:
    if intrinsics is not None and intrinsics.ndim == 1 and intrinsics.size == 16:
        intrinsics = intrinsics.reshape((4, 4))
    if intrinsics is None or intrinsics.shape not in {(3, 3), (4, 4)} or not np.isfinite(intrinsics).all():
        return None
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if abs(fx) <= 1e-12 or abs(fy) <= 1e-12:
        return None
    return [
        round((x - cx) * depth / fx, 6),
        round((y - cy) * depth / fy, 6),
        round(depth, 6),
    ]


def extract_marker_geometry(image: Any, depth: Any, intrinsics: Any = None) -> dict[str, Any]:
    """Extract red/blue marker evidence without consulting question answers."""
    image_array = np.asarray(image)
    depth_array = _array(depth)
    if image_array.ndim != 3 or depth_array is None or depth_array.ndim != 2:
        return {}
    image_height, image_width = image_array.shape[:2]
    depth_height, depth_width = depth_array.shape[:2]
    intrinsic_array = _array(intrinsics)
    markers: dict[str, Any] = {}
    for color in ("red", "green", "blue"):
        mask = _marker_mask(image_array, color)
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        visible_x, visible_y = float(np.median(xs)), float(np.median(ys))
        image_x = (min_x + max_x) / 2.0
        image_y = (min_y + max_y) / 2.0
        depth_x = image_x * depth_width / max(image_width, 1)
        depth_y = image_y * depth_height / max(image_height, 1)
        marker_depth, sample_count = _sample_depth(depth_array, depth_x, depth_y)
        if marker_depth is None:
            continue
        markers[color] = {
            "pixel_xy": [round(image_x, 3), round(image_y, 3)],
            "visible_pixel_median_xy": [round(visible_x, 3), round(visible_y, 3)],
            "pixel_bbox": [min_x, min_y, max_x, max_y],
            "depth_pixel_xy": [round(depth_x, 3), round(depth_y, 3)],
            "marker_pixel_count": int(xs.size),
            "depth_sample_count": sample_count,
            "depth": marker_depth,
            "camera_xyz": _camera_xyz(depth_x, depth_y, marker_depth, intrinsic_array),
            "unit": "dataset_native",
        }
    return markers


def _marker_relations(markers: Mapping[str, Any]) -> dict[str, Any]:
    pairwise: dict[str, Any] = {}
    marker_items = [
        (str(color), marker)
        for color, marker in markers.items()
        if isinstance(marker, Mapping)
    ]
    for left_index, (left_color, left_marker) in enumerate(marker_items):
        left_xyz = left_marker.get("camera_xyz")
        left_depth = left_marker.get("depth")
        if not isinstance(left_xyz, list) or len(left_xyz) != 3:
            continue
        for right_color, right_marker in marker_items[left_index + 1 :]:
            right_xyz = right_marker.get("camera_xyz")
            right_depth = right_marker.get("depth")
            if not isinstance(right_xyz, list) or len(right_xyz) != 3:
                continue
            pair_name = f"{left_color}_to_{right_color}"
            relation: dict[str, Any] = {
                "euclidean_distance": round(
                    math.sqrt(
                        sum((float(a) - float(b)) ** 2 for a, b in zip(left_xyz, right_xyz))
                    ),
                    6,
                ),
                "unit": "dataset_native",
            }
            if isinstance(left_depth, (int, float)) and isinstance(right_depth, (int, float)):
                relation["depth_difference_right_minus_left"] = round(
                    float(right_depth) - float(left_depth), 6
                )
            pairwise[pair_name] = relation

    red = markers.get("red")
    blue = markers.get("blue")
    if not isinstance(red, Mapping) or not isinstance(blue, Mapping):
        return {"pairwise": pairwise} if pairwise else {}
    relation = {
        "depth_difference_blue_minus_red": round(float(blue["depth"]) - float(red["depth"]), 6),
        "unit": "dataset_native",
    }
    red_xyz, blue_xyz = red.get("camera_xyz"), blue.get("camera_xyz")
    if isinstance(red_xyz, list) and isinstance(blue_xyz, list):
        relation["euclidean_distance"] = round(
            math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(red_xyz, blue_xyz))),
            6,
        )
    if pairwise:
        relation["pairwise"] = pairwise
    return relation


def _depth_stats(value: Any) -> dict[str, Any]:
    array = _array(value)
    if array is None:
        return {"valid": False, "shape": None, "valid_pixel_count": 0}
    finite = array[np.isfinite(array)]
    finite = finite[finite > 0]
    if finite.size == 0:
        return {
            "valid": False,
            "shape": list(array.shape),
            "valid_pixel_count": 0,
        }
    return {
        "valid": True,
        "shape": list(array.shape),
        "valid_pixel_count": int(finite.size),
        "min": round(float(finite.min()), 6),
        "max": round(float(finite.max()), 6),
        "mean": round(float(finite.mean()), 6),
        "median": round(float(np.median(finite)), 6),
        "p10": round(float(np.percentile(finite, 10)), 6),
        "p90": round(float(np.percentile(finite, 90)), 6),
    }


def _view_value(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def _has_answer_free_reference(row: Mapping[str, Any]) -> bool:
    reference_keys = (
        "points",
        "point",
        "point_annotations",
        "objects",
        "object_annotations",
        "regions",
    )
    return any(row.get(key) not in (None, [], {}) for key in reference_keys)


def required_marker_colors(question: str) -> set[str]:
    """Return marker colors explicitly referenced by a SPAR question."""
    return {
        color
        for color in ("red", "green", "blue")
        if re.search(rf"\b{color}\s+(?:point|bbox)\b", str(question), re.IGNORECASE)
    }


def _task_status(
    row: Mapping[str, Any], views: list[dict[str, Any]], question: Mapping[str, Any]
) -> tuple[str, str | None]:
    task = str(row.get("task", "")).strip().lower()
    has_markers = any(view.get("markers") for view in views)
    required = required_marker_colors(str(question.get("question", "")))
    available = set().union(*(set(view.get("markers", {})) for view in views))
    if required and not required.issubset(available):
        return "geometry_unavailable", "required_marker_missing"
    if (task.endswith("_oc") or task.endswith("_oo")) and not (
        _has_answer_free_reference(row) or has_markers
    ):
        return "geometry_unavailable", "no_point_or_object_reference"
    return "available", None


def contains_answer_leak(value: Any) -> bool:
    """Detect answer-bearing keys in a geometry payload recursively."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in ("answer", "gold", "correct_label")):
                return True
            if contains_answer_leak(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(contains_answer_leak(item) for item in value)
    return False


def build_geometry_record(row: Mapping[str, Any], question_record: Mapping[str, Any]) -> dict[str, Any]:
    images = to_list(row.get("image"))
    exported_images = to_list(question_record.get("images"))
    depths = to_list(row.get("depth"))
    poses = to_list(row.get("pose"))
    color_intrinsics = to_list(row.get("intrinsic_color"))
    depth_intrinsics = to_list(row.get("intrinsic_depth"))
    views = []
    for index, image in enumerate(images):
        depth_value = _view_value(depths, index)
        intrinsic_depth_value = _view_value(depth_intrinsics, index)
        markers = extract_marker_geometry(image, depth_value, intrinsic_depth_value)
        views.append(
            {
                "view_index": index,
                "image_name": str(
                    _view_value(exported_images, index)
                    or getattr(image, "filename", None)
                    or getattr(image, "name", None)
                    or image
                ),
                "depth": _depth_stats(depth_value),
                "intrinsic_color": validate_intrinsics(_view_value(color_intrinsics, index)),
                "intrinsic_depth": validate_intrinsics(intrinsic_depth_value),
                "pose": validate_pose(_view_value(poses, index)),
                "markers": markers,
                "marker_relations": _marker_relations(markers),
            }
        )
    required_colors = sorted(required_marker_colors(str(question_record.get("question", ""))))
    task_status, unavailable_reason = _task_status(row, views, question_record)
    record: dict[str, Any] = {
        "question_id": question_record.get("question_id"),
        "dataset": "spar_bench_tiny_rgbd",
        "source_id": str(row.get("id", "unknown")),
        "geometry_source": "oracle_rgbd",
        "task": str(row.get("task", "")),
        "task_geometry_status": task_status,
        "required_marker_colors": required_colors,
        "views": views,
        "view_count": len(views),
        "available_view_count": sum(
            bool(view["depth"].get("valid")) for view in views
        ),
    }
    if unavailable_reason:
        record["unavailable_reason"] = unavailable_reason
    return record


def build_geometry_files(
    rows: list[dict[str, Any]],
    question_records: list[dict[str, Any]],
    output_dir: Path,
    overwrite: bool = False,
) -> list[Path]:
    by_source = {str(row.get("id")): row for row in rows}
    paths = []
    for question_record in question_records:
        row = by_source.get(str(question_record.get("source_id")))
        if row is None:
            raise ValueError(f"No source row for {question_record.get('question_id')}")
        record = build_geometry_record(row, question_record)
        if contains_answer_leak(record):
            raise ValueError(f"Answer leak detected for {question_record.get('question_id')}")
        path = output_dir / f"{question_record['question_id']}.json"
        write_json(record, path, overwrite=overwrite)
        paths.append(path)
    return paths


def build_geometry_files_streaming(
    dataset_name: str,
    split: str,
    question_records: list[dict[str, Any]],
    output_dir: Path,
    streaming: bool = True,
    overwrite: bool = False,
) -> list[Path]:
    """Read source rows one at a time and write geometry only for selected IDs."""
    questions_by_source = {
        str(question.get("source_id")): question for question in question_records
    }
    written: dict[str, Path] = {}
    for row in iter_dataset_rows(
        dataset_name=dataset_name,
        split=split,
        streaming=streaming,
        columns=[
            "id",
            "image",
            "depth",
            "pose",
            "intrinsic_color",
            "intrinsic_depth",
            "task",
        ],
    ):
        source_id = str(row.get("id"))
        question = questions_by_source.get(source_id)
        if question is None:
            continue
        record = build_geometry_record(row, question)
        if contains_answer_leak(record):
            raise ValueError(f"Answer leak detected for {question.get('question_id')}")
        path = output_dir / f"{question['question_id']}.json"
        write_json(record, path, overwrite=overwrite)
        written[source_id] = path

    missing = set(questions_by_source) - set(written)
    if missing:
        raise ValueError(f"Selected source rows were not available: {sorted(missing)}")
    return [written[str(question["source_id"])] for question in question_records]


def repair_geometry_file(path: Path) -> None:
    """Repair marker camera coordinates in an already-generated geometry file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    for view in payload.get("views", []):
        intrinsic = _array(view.get("intrinsic_depth", {}).get("values"))
        if intrinsic is not None and intrinsic.ndim == 1 and intrinsic.size == 16:
            intrinsic = intrinsic.reshape((4, 4))
        markers = view.get("markers", {})
        if not isinstance(markers, Mapping):
            continue
        for marker in markers.values():
            if not isinstance(marker, Mapping):
                continue
            pixel = marker.get("depth_pixel_xy")
            depth = marker.get("depth")
            if isinstance(pixel, list) and len(pixel) == 2 and isinstance(depth, (int, float)):
                marker["camera_xyz"] = _camera_xyz(
                    float(pixel[0]), float(pixel[1]), float(depth), intrinsic
                )
        view["marker_relations"] = _marker_relations(markers)
    write_json(payload, path, overwrite=True)
