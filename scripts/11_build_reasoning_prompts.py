"""Build reasoning prompt records from benchmark questions and geometry JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def geometry_path_for_image(geometry_dir: Path, image_name: str) -> Path:
    return geometry_dir / f"{Path(image_name).stem}.json"


def format_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    return str(value)


def labels_in_geometry(geometry: dict[str, Any]) -> set[str]:
    labels = set()
    for obj in geometry.get("objects", []):
        if isinstance(obj, dict) and isinstance(obj.get("label"), str):
            labels.add(obj["label"])
    return labels


def missing_target_objects(geometry: dict[str, Any], target_objects: list[str]) -> list[str]:
    labels = labels_in_geometry(geometry)
    return [label for label in target_objects if label not in labels]


def object_line(obj: dict[str, Any]) -> str:
    object_id = format_value(obj.get("object_id"))
    label = format_value(obj.get("label"))
    position = f"{format_value(obj.get('horizontal_position'))}/{format_value(obj.get('vertical_position'))}"
    centroid = format_value(obj.get("mask_centroid"))
    area = format_value(obj.get("mask_area_fraction"))
    depth_median = format_value(obj.get("relative_depth_median"))
    depth_percentile = format_value(obj.get("relative_depth_percentile"))
    depth_hint = format_value(obj.get("depth_order_hint"))
    raw_label = obj.get("raw_label")
    raw_suffix = f", raw_label={raw_label}" if raw_label and raw_label != obj.get("label") else ""
    return (
        f"- {object_id} {label}: position={position}, centroid={centroid}, "
        f"area_fraction={area}, median_depth={depth_median}, "
        f"depth_percentile={depth_percentile}, depth_hint={depth_hint}{raw_suffix}"
    )


def relation_line(relation: dict[str, Any]) -> str:
    return (
        f"- {format_value(relation.get('object_a'))} {format_value(relation.get('label_a'))} "
        f"vs {format_value(relation.get('object_b'))} {format_value(relation.get('label_b'))}: "
        f"horizontal={format_value(relation.get('horizontal_relation'))}, "
        f"vertical={format_value(relation.get('vertical_relation'))}, "
        f"depth={format_value(relation.get('depth_relation'))}"
    )


def summarize_geometry(geometry: dict[str, Any], target_objects: list[str]) -> str:
    lines = [
        f"Image: {format_value(geometry.get('image'))}",
        f"Image size: {format_value(geometry.get('image_width'))} x {format_value(geometry.get('image_height'))}",
        f"Depth assumption: {format_value(geometry.get('depth_order_assumption'))}",
        f"Target object labels: {', '.join(target_objects) if target_objects else 'none'}",
        "Objects:",
    ]

    objects = geometry.get("objects", [])
    if isinstance(objects, list) and objects:
        lines.extend(object_line(obj) for obj in objects if isinstance(obj, dict))
    else:
        lines.append("- none detected")

    lines.append("Pairwise relations:")
    relations = geometry.get("pairwise_relations", [])
    if isinstance(relations, list) and relations:
        lines.extend(relation_line(relation) for relation in relations if isinstance(relation, dict))
    else:
        lines.append("- none available")

    missing = missing_target_objects(geometry, target_objects)
    if missing:
        lines.append(f"Missing target labels in geometry: {', '.join(missing)}")

    return "\n".join(lines)


def answer_instruction() -> str:
    return "Answer with the object label only. If the evidence is insufficient, answer unknown."


def build_pure_vlm_prompt(question: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are answering a visual spatial reasoning question from the image.",
            "Use only the image content. Do not use external assumptions.",
            f"Question: {question['question']}",
            answer_instruction(),
        ]
    )


def build_geometry_llm_prompt(question: dict[str, Any], geometry_summary: str) -> str:
    return "\n".join(
        [
            "You are answering a spatial reasoning question using object-level geometry extracted from the image.",
            "The geometry comes from object detection, SAM2 masks, and monocular relative depth.",
            "Use the listed objects, positions, mask areas, depth statistics, and pairwise relations.",
            geometry_summary,
            f"Question: {question['question']}",
            answer_instruction(),
        ]
    )


def build_geovlm_prompt(question: dict[str, Any], geometry_summary: str) -> str:
    return "\n".join(
        [
            "You are answering a visual spatial reasoning question using both the image and object-level geometry.",
            "Use the image as primary evidence and use the geometry as structured support.",
            "If the image and geometry disagree, prefer visible image evidence and mention uncertainty only internally.",
            geometry_summary,
            f"Question: {question['question']}",
            answer_instruction(),
        ]
    )


def build_prompt_record(question: dict[str, Any], geometry: dict[str, Any], geometry_path: Path) -> dict[str, Any]:
    target_objects = question.get("target_objects", [])
    if not isinstance(target_objects, list):
        target_objects = []
    target_objects = [str(item) for item in target_objects]
    geometry_summary = summarize_geometry(geometry, target_objects=target_objects)
    evaluation = question.get("evaluation", {})
    acceptable_answers = evaluation.get("acceptable_answers", []) if isinstance(evaluation, dict) else []

    return {
        "question_id": question.get("question_id"),
        "image": question.get("image"),
        "question": question.get("question"),
        "type": question.get("type"),
        "target_objects": target_objects,
        "answer": question.get("answer"),
        "acceptable_answers": acceptable_answers,
        "geometry_path": normalized_path(geometry_path),
        "missing_target_objects": missing_target_objects(geometry, target_objects),
        "pure_vlm_prompt": build_pure_vlm_prompt(question),
        "geometry_llm_prompt": build_geometry_llm_prompt(question, geometry_summary),
        "geovlm_prompt": build_geovlm_prompt(question, geometry_summary),
    }


def build_prompts_file(question_path: Path, geometry_dir: Path, output_path: Path, overwrite: bool) -> int:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Prompt output already exists: {output_path}")

    payload = load_json(question_path)
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"Question file must contain a questions list: {question_path}")

    records = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        image_name = question.get("image")
        if not isinstance(image_name, str):
            raise ValueError(f"Question missing image field: {question}")
        geometry_path = geometry_path_for_image(geometry_dir, image_name)
        if not geometry_path.exists():
            raise FileNotFoundError(f"Geometry file not found for {image_name}: {geometry_path}")
        geometry = load_json(geometry_path)
        records.append(build_prompt_record(question=question, geometry=geometry, geometry_path=geometry_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GeoVLM reasoning prompts from questions and geometry.")
    parser.add_argument("--questions", type=Path, default=Path("data/questions.json"))
    parser.add_argument("--geometry-dir", type=Path, default=Path("outputs/geometry"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reasoning/prompts.jsonl"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prompt JSONL output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        count = build_prompts_file(
            question_path=args.questions,
            geometry_dir=args.geometry_dir,
            output_path=args.output,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print(f"Wrote {args.output} ({count} prompt records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
