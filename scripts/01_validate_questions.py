"""Validate GeoVLM-SceneReasoner benchmark question files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_QUESTION_FIELDS = {
    "question_id",
    "image",
    "question",
    "type",
    "target_objects",
    "answer",
    "evaluation",
}

ALLOWED_TYPES = {
    "closer_farther",
    "left_right",
    "front_back",
    "occlusion",
    "support_relation",
    "physical_size",
}


def load_questions(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_question_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    questions = payload.get("questions")
    if not isinstance(questions, list):
        return ["top-level field 'questions' must be a list"]

    seen_ids: set[str] = set()
    for index, question in enumerate(questions):
        prefix = f"questions[{index}]"
        if not isinstance(question, dict):
            errors.append(f"{prefix} must be an object")
            continue

        for field in sorted(REQUIRED_QUESTION_FIELDS):
            if field not in question:
                errors.append(f"{prefix} missing required field: {field}")

        question_id = question.get("question_id")
        if isinstance(question_id, str):
            if question_id in seen_ids:
                errors.append(f"{prefix} duplicate question_id: {question_id}")
            seen_ids.add(question_id)

        question_type = question.get("type")
        if question_type is not None and question_type not in ALLOWED_TYPES:
            allowed = ", ".join(sorted(ALLOWED_TYPES))
            errors.append(f"{prefix} unknown type: {question_type}. Allowed: {allowed}")

        target_objects = question.get("target_objects")
        if target_objects is not None:
            if not isinstance(target_objects, list) or not all(isinstance(item, str) for item in target_objects):
                errors.append(f"{prefix} target_objects must be a list of strings")

        evaluation = question.get("evaluation")
        if evaluation is not None and not isinstance(evaluation, dict):
            errors.append(f"{prefix} evaluation must be an object")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GeoVLM-SceneReasoner question JSON.")
    parser.add_argument("--questions", type=Path, default=Path("data/questions.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.questions.exists():
        print(f"Question file not found: {args.questions}")
        print("Start from data/questions.example.json and save your benchmark as data/questions.json.")
        return 1

    payload = load_questions(args.questions)
    errors = validate_question_payload(payload)
    if errors:
        print("Question validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    question_count = len(payload.get("questions", []))
    print(f"Question validation passed: {question_count} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
