"""Expand scene-level question templates across GeoVLM image views."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        (path for path in image_dir.iterdir() if is_image_file(path)),
        key=lambda path: path.name.lower(),
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_camera_variation(image_path: Path, scene_group: str) -> str:
    pattern = rf"^{re.escape(scene_group)}_(view_\d+)$"
    match = re.match(pattern, image_path.stem)
    if match:
        return match.group(1)
    return image_path.stem


def validate_template_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload.get("scene_group"), str):
        errors.append("top-level field 'scene_group' must be a string")

    templates = payload.get("templates")
    if not isinstance(templates, list) or not templates:
        errors.append("top-level field 'templates' must be a non-empty list")
        return errors

    required_fields = {
        "template_id",
        "question",
        "type",
        "target_objects",
        "answer",
        "evaluation",
    }
    for index, template in enumerate(templates):
        prefix = f"templates[{index}]"
        if not isinstance(template, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field in sorted(required_fields):
            if field not in template:
                errors.append(f"{prefix} missing required field: {field}")
    return errors


def expand_question_templates(images: list[Path], template_payload: dict[str, Any]) -> dict[str, Any]:
    errors = validate_template_payload(template_payload)
    if errors:
        raise ValueError("; ".join(errors))

    scene_group = template_payload["scene_group"]
    questions: list[dict[str, Any]] = []

    for image in images:
        camera_variation = infer_camera_variation(image, scene_group)
        for template in template_payload["templates"]:
            question = deepcopy(template)
            template_id = question.pop("template_id")
            question["question_id"] = f"{image.stem}_{template_id}"
            question["image"] = image.name
            question["scene_group"] = scene_group
            question["camera_variation"] = camera_variation
            question["template_id"] = template_id
            questions.append(question)

    return {
        "version": template_payload.get("version", "0.1"),
        "description": template_payload.get(
            "description",
            "Draft GeoVLM-SceneReasoner benchmark questions expanded from templates.",
        ),
        "scene_group": scene_group,
        "questions": questions,
    }


def write_questions(payload: dict[str, Any], output_path: Path, force: bool = False) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark questions from scene-level templates.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--templates", type=Path, default=Path("data/question_templates.example.json"))
    parser.add_argument("--output", type=Path, default=Path("data/questions.draft.json"))
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = collect_image_files(args.image_dir)
    if not images:
        print(f"No image files found in {args.image_dir}")
        return 1

    template_payload = load_json(args.templates)
    payload = expand_question_templates(images, template_payload)

    try:
        write_questions(payload, args.output, force=args.force)
    except FileExistsError as error:
        print(error)
        print("Use --force to overwrite it.")
        return 1

    print(f"Images: {len(images)}")
    print(f"Templates: {len(template_payload['templates'])}")
    print(f"Questions written: {len(payload['questions'])}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
