"""Build metric-aware, answer-free prompts for the public SPAR tracks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from scripts.public_spar_common import json_safe, write_json


def _format_geometry(geometry: Mapping[str, Any]) -> str:
    lines = [
        "Structured oracle RGB-D evidence:",
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
            for color in ("red", "blue"):
                marker = markers.get(color)
                if isinstance(marker, Mapping):
                    lines.append(
                        f"View {view.get('view_index')} {color} marker: "
                        f"pixel_xy={marker.get('pixel_xy')}, depth={marker.get('depth')}, "
                        f"camera_xyz={marker.get('camera_xyz')}, unit={marker.get('unit', 'dataset_native')}"
                    )
        relations = view.get("marker_relations", {})
        if isinstance(relations, Mapping) and relations:
            lines.append(
                f"View {view.get('view_index')} marker relations: "
                f"depth_difference_blue_minus_red={relations.get('depth_difference_blue_minus_red', 'unknown')}, "
                f"euclidean_distance={relations.get('euclidean_distance', 'unknown')}, "
                f"unit={relations.get('unit', 'dataset_native')}"
            )
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


def _question_text(question: Mapping[str, Any]) -> str:
    return str(question.get("question", "")).strip()


def build_prompt_record(question: Mapping[str, Any], geometry: Mapping[str, Any]) -> dict[str, Any]:
    geometry_text = _format_geometry(geometry)
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
            geometry_text,
            f"Question: {question_text}",
            instruction,
        ]
    )
    geovlm = "\n".join(
        [
            "You are answering a spatial reasoning benchmark question using the RGB image and structured RGB-D geometry.",
            "Use the image for visual semantics and the RGB-D evidence for calibrated depth, distance, pose, and spatial reasoning.",
            "For metric depth or distance questions, prefer valid calibrated geometry over visual scale shortcuts.",
            geometry_text,
            f"Question: {question_text}",
            instruction,
        ]
    )
    return {
        "question_id": question.get("question_id"),
        "dataset": question.get("dataset"),
        "source_id": question.get("source_id"),
        "images": json_safe(question.get("images", [])),
        "task": question.get("task"),
        "format_type": question.get("format_type"),
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
