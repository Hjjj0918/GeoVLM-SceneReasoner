"""Build metric-aware, answer-free prompts for the public SPAR tracks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from scripts.public_spar_common import json_safe, write_json


def _format_geometry(geometry: Mapping[str, Any], task: str = "") -> str:
    spatial_relation = task.strip().lower() == "obj_spatial_relation_oo"
    lines = [
        "Structured oracle RGB-D evidence:",
        "Coordinate convention: camera_xyz is (x, y, z); x increases to the right, y increases downward, and z increases farther from the camera.",
        f"Geometry status: {geometry.get('task_geometry_status', 'unknown')}",
        f"View count: {geometry.get('view_count', len(geometry.get('views', [])))}",
    ]
    for view in geometry.get("views", []):
        if not isinstance(view, Mapping):
            continue
        depth = view.get("depth", {})
        lines.append(
            f"View {view.get('view_index')}: depth_valid={depth.get('valid', False)}, "
            f"valid_pixels={depth.get('valid_pixel_count', 0)}, "
            f"depth_min={depth.get('min', 'unknown')}, depth_max={depth.get('max', 'unknown')}, "
            f"depth_median={depth.get('median', 'unknown')}, "
            f"color_intrinsics_valid={view.get('intrinsic_color', {}).get('valid', False)}, "
            f"depth_intrinsics_valid={view.get('intrinsic_depth', {}).get('valid', False)}, "
            f"pose_valid={view.get('pose', {}).get('valid', False)}"
        )
        markers = view.get("markers", {})
        if isinstance(markers, Mapping):
            color_order = {"red": 0, "green": 1, "blue": 2}
            marker_items = sorted(
                markers.items(),
                key=lambda item: (color_order.get(str(item[0]), 99), str(item[0])),
            )
            for color, marker in marker_items:
                if isinstance(marker, Mapping):
                    lines.append(
                        f"View {view.get('view_index')} {color} marker: "
                        f"pixel_xy={marker.get('pixel_xy')}, pixel_bbox={marker.get('pixel_bbox')}, "
                        f"visible_pixel_median_xy={marker.get('visible_pixel_median_xy')}, "
                        f"depth={marker.get('depth')}, "
                        f"camera_xyz={marker.get('camera_xyz')}, unit={marker.get('unit', 'dataset_native')}"
                    )
        relations = view.get("marker_relations", {})
        if isinstance(relations, Mapping) and relations:
            relation_line = (
                f"View {view.get('view_index')} marker relations: "
                f"depth_difference_blue_minus_red={relations.get('depth_difference_blue_minus_red', 'unknown')}"
            )
            if not spatial_relation:
                relation_line += f", euclidean_distance={relations.get('euclidean_distance', 'unknown')}"
            lines.append(relation_line + f", unit={relations.get('unit', 'dataset_native')}")
            pairwise = relations.get("pairwise", {})
            if isinstance(pairwise, Mapping):
                for pair_name, pair in pairwise.items():
                    if not isinstance(pair, Mapping):
                        continue
                    pair_line = (
                        f"View {view.get('view_index')} marker pair {pair_name}: "
                        f"depth_difference={pair.get('depth_difference_right_minus_left', 'unknown')}"
                    )
                    if not spatial_relation:
                        pair_line += f", euclidean_distance={pair.get('euclidean_distance', 'unknown')}"
                    lines.append(pair_line + f", unit={pair.get('unit', 'dataset_native')}")
    return "\n".join(lines)


def _instruction(question: Mapping[str, Any]) -> str:
    evaluation = question.get("evaluation", {})
    metric = evaluation.get("metric") if isinstance(evaluation, Mapping) else "exact_match"
    if metric == "multiple_choice":
        return "Return only the canonical multiple-choice label. Do not explain."
    if metric == "numeric_tolerance":
        unit = evaluation.get("unit", "meter") if isinstance(evaluation, Mapping) else "meter"
        return f"Return exactly one numeric value in {unit}; do not include explanation or extra numbers."
    return "Return the shortest exact matching phrase only; do not explain."


def _task_geometry_instruction(question: Mapping[str, Any]) -> str:
    task = str(question.get("task", "")).strip().lower()
    if task == "distance_infer_center_oo":
        return (
            "For center-distance questions, compare the full 3D Euclidean distance in camera_xyz from each candidate marker "
            "to the reference marker; do not compare candidate markers to each other.\n"
            "Do not use z alone, depth alone, or image position as a substitute for the full 3D distance."
        )
    if task == "obj_spatial_relation_oo":
        return "\n".join(
            [
                "For object spatial-relation questions, interpret closer/farther relative to the observer using camera-space z: larger z is farther and smaller z is closer.",
                "Do not use Euclidean distance between the two markers for the closer/farther relation.",
                "Use image-space bbox centers for left/right and above/below; larger image y is lower.",
                "Report above/below from the image-space bbox centers; leave that relation empty only when the centers are indistinguishable or the evidence is ambiguous.",
            ]
        )
    return "Use only the coordinate convention and relations explicitly relevant to the question."


def _task_closing_instruction(question: Mapping[str, Any], geometry: Mapping[str, Any]) -> str:
    task = str(question.get("task", "")).strip().lower()
    if task == "distance_infer_center_oo":
        distances: list[tuple[str, float]] = []
        views = geometry.get("views", [])
        if isinstance(views, list):
            for view in views:
                if not isinstance(view, Mapping):
                    continue
                relations = view.get("marker_relations", {})
                pairwise = relations.get("pairwise", {}) if isinstance(relations, Mapping) else {}
                if not isinstance(pairwise, Mapping):
                    continue
                for pair_name in ("red_to_green", "red_to_blue"):
                    pair = pairwise.get(pair_name)
                    if not isinstance(pair, Mapping):
                        continue
                    distance = pair.get("euclidean_distance")
                    if isinstance(distance, (int, float)):
                        distances.append((pair_name, float(distance)))
        if distances:
            distances.sort(key=lambda item: item[1])
            closer_name, closer_distance = distances[0]
            farther_name, farther_distance = distances[-1]
            direction = {
                "red_to_green": "green is farther than blue",
                "red_to_blue": "blue is farther than green",
            }.get(farther_name, "choose the larger red-to-candidate distance")
            decision_aid = (
                "Decision aid: "
                f"{closer_name}={closer_distance}; {farther_name}={farther_distance}; "
                f"{direction}."
            )
        else:
            decision_aid = "Decision aid: compare the two red-to-candidate Euclidean distances and choose the larger one."
        return decision_aid
    if task == "obj_spatial_relation_oo":
        question_text = _question_text(question)
        colors = []
        for match in re.finditer(r"\b(red|green|blue)\b", question_text, re.IGNORECASE):
            color = match.group(1).lower()
            if color not in colors:
                colors.append(color)
        if len(colors) >= 2:
            left_color, right_color = colors[0], colors[1]
            view = geometry.get("views", [])
            marker_map: Mapping[str, Any] | None = None
            if isinstance(view, list):
                for item in view:
                    if isinstance(item, Mapping) and isinstance(item.get("markers"), Mapping):
                        marker_map = item["markers"]
                        break
            if isinstance(marker_map, Mapping):
                left_marker = marker_map.get(left_color)
                right_marker = marker_map.get(right_color)
                if isinstance(left_marker, Mapping) and isinstance(right_marker, Mapping):
                    left_xy = left_marker.get("pixel_xy")
                    right_xy = right_marker.get("pixel_xy")
                    left_xyz = left_marker.get("camera_xyz")
                    right_xyz = right_marker.get("camera_xyz")
                    pieces: list[str] = []
                    if (
                        isinstance(left_xy, list)
                        and isinstance(right_xy, list)
                        and len(left_xy) >= 2
                        and len(right_xy) >= 2
                    ):
                        left_x, left_y = float(left_xy[0]), float(left_xy[1])
                        right_x, right_y = float(right_xy[0]), float(right_xy[1])
                        if abs(left_x - right_x) > 1e-6:
                            pieces.append(
                                f"{left_color} is {'left' if left_x < right_x else 'right'} of {right_color}"
                            )
                        if abs(left_y - right_y) > 1e-6:
                            pieces.append(
                                f"{left_color} is {'above' if left_y < right_y else 'below'} {right_color}"
                            )
                    if (
                        isinstance(left_xyz, list)
                        and isinstance(right_xyz, list)
                        and len(left_xyz) >= 3
                        and len(right_xyz) >= 3
                    ):
                        left_z, right_z = float(left_xyz[2]), float(right_xyz[2])
                        pieces.append(
                            f"{left_color} is {'farther' if left_z > right_z else 'closer'} than {right_color}"
                        )
                    if pieces:
                        return "Decision aid: " + "; ".join(pieces) + "."
        return "Decision aid: use the question colors, compare bbox-center x/y for left/right and above/below, and compare camera-space z for closer/farther."
    return ""


def _question_text(question: Mapping[str, Any]) -> str:
    return str(question.get("question", "")).strip()


def build_prompt_record(question: Mapping[str, Any], geometry: Mapping[str, Any]) -> dict[str, Any]:
    task = str(question.get("task", ""))
    geometry_text = _format_geometry(geometry, task=task)
    task_geometry_instruction = _task_geometry_instruction(question)
    closing_instruction = _task_closing_instruction(question, geometry)
    question_text = _question_text(question)
    instruction = _instruction(question)
    pure = "\n".join(
        [
            "You are answering a spatial reasoning benchmark question from the RGB image.",
            "Use only the image and the question. Do not use depth, camera pose, intrinsics, or external assumptions.",
            f"Question: {question_text}",
            instruction,
        ]
    )
    geometry_only = "\n".join(
        [
            "You are answering a spatial reasoning benchmark question using structured RGB-D geometry.",
            "Use only the structured evidence below and the question. Do not infer unavailable visual appearance.",
            task_geometry_instruction,
            geometry_text,
            f"Question: {question_text}",
            instruction,
        ]
    )
    geovlm_parts = [
        closing_instruction,
        "You are answering a spatial reasoning benchmark question using the RGB image and structured RGB-D geometry.",
        "Use the image for visual semantics and the RGB-D evidence for calibrated depth, distance, pose, and spatial reasoning.",
        "For all depth, distance, and spatial-relation questions, treat camera_xyz geometry as authoritative.",
        "Use the image only to identify which visual marker corresponds to red, green, or blue.",
        "Do not override geometric relations based on apparent image position or object appearance.",
        task_geometry_instruction,
        geometry_text,
        f"Question: {question_text}",
        instruction,
    ]
    geovlm = "\n".join(
        [part for part in geovlm_parts if part]
    )
    return {
        "question_id": question.get("question_id"),
        "dataset": question.get("dataset"),
        "source_id": question.get("source_id"),
        "images": json_safe(question.get("images", [])),
        "task": question.get("task"),
        "format_type": question.get("format_type"),
        "img_type": question.get("img_type"),
        "evaluation": json_safe(question.get("evaluation", {})),
        "answer": json_safe(question.get("answer")),
        "geometry_path": question.get("geometry_path"),
        "geometry_source": question.get("geometry_source"),
        "requires_image": {"pure_vlm": True, "geometry_only": False, "geovlm": True},
        "prompts": {"pure_vlm": pure, "geometry_only": geometry_only, "geovlm": geovlm},
    }


def prompt_contains_answer_leak(prompts: Mapping[str, str], answer: Any) -> bool:
    answer_text = str(answer).strip().lower()
    if not answer_text:
        return False
    explicit_key = re.compile(r"(?:gold|correct|reference)\s+answer\s*:", re.IGNORECASE)
    return any(explicit_key.search(str(prompt)) for prompt in prompts.values())


def build_prompts_file(
    questions: list[dict[str, Any]],
    geometries: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    overwrite: bool = False,
) -> int:
    records = []
    for question in questions:
        question_id = str(question.get("question_id"))
        geometry = geometries.get(question_id)
        if geometry is None:
            raise ValueError(f"Missing geometry for {question_id}")
        record = build_prompt_record(question, geometry)
        if prompt_contains_answer_leak(record["prompts"], question.get("answer")):
            raise ValueError(f"Answer leak detected in prompts for {question_id}")
        records.append(record)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        import json

        for record in records:
            file.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
    return len(records)
