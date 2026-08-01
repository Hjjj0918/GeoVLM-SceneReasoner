"""Run a simple geometry-only rule baseline over GeoVLM prompt records."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


SUPPORTED_TYPES = {"closer_farther", "physical_size", "support_relation"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if isinstance(record, dict):
                records.append(record)
    return records


def normalized_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def object_by_label(geometry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    objects = {}
    for obj in geometry.get("objects", []):
        if isinstance(obj, dict) and isinstance(obj.get("label"), str):
            objects[obj["label"]] = obj
    return objects


def find_pairwise_relation(geometry: dict[str, Any], label_a: str, label_b: str) -> tuple[dict[str, Any] | None, bool]:
    for relation in geometry.get("pairwise_relations", []):
        if not isinstance(relation, dict):
            continue
        if relation.get("label_a") == label_a and relation.get("label_b") == label_b:
            return relation, False
        if relation.get("label_a") == label_b and relation.get("label_b") == label_a:
            return relation, True
    return None, False


def invert_depth_relation(relation: str) -> str:
    if relation == "closer_than":
        return "farther_than"
    if relation == "farther_than":
        return "closer_than"
    return relation


def numeric_value(obj: dict[str, Any], field: str) -> float | None:
    value = obj.get(field)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def answer_from_depth_relation(geometry: dict[str, Any], target_objects: list[str]) -> tuple[str, str, float]:
    label_a, label_b = target_objects[:2]
    relation, reversed_order = find_pairwise_relation(geometry, label_a, label_b)
    if relation:
        depth_relation = str(relation.get("depth_relation", "unknown"))
        if reversed_order:
            depth_relation = invert_depth_relation(depth_relation)
        if depth_relation == "closer_than":
            return label_a, "pairwise_depth_relation", 0.9
        if depth_relation == "farther_than":
            return label_b, "pairwise_depth_relation", 0.9

    objects = object_by_label(geometry)
    depth_a = numeric_value(objects[label_a], "relative_depth_median")
    depth_b = numeric_value(objects[label_b], "relative_depth_median")
    if depth_a is None or depth_b is None:
        return "unknown", "relative_depth_median", 0.0
    if abs(depth_a - depth_b) <= 1e-6:
        return "unknown", "relative_depth_median", 0.0

    assumption = geometry.get("depth_order_assumption", "higher_relative_depth_is_closer")
    higher_is_closer = assumption != "lower_relative_depth_is_closer"
    if higher_is_closer:
        return (label_a, "relative_depth_median", 0.7) if depth_a > depth_b else (label_b, "relative_depth_median", 0.7)
    return (label_a, "relative_depth_median", 0.7) if depth_a < depth_b else (label_b, "relative_depth_median", 0.7)


def answer_from_area(geometry: dict[str, Any], target_objects: list[str]) -> tuple[str, str, float]:
    objects = object_by_label(geometry)
    best_label = "unknown"
    best_area = -1.0
    for label in target_objects:
        area = numeric_value(objects[label], "mask_area_fraction")
        if area is not None and area > best_area:
            best_label = label
            best_area = area
    confidence = 0.7 if best_label != "unknown" else 0.0
    return best_label, "mask_area_fraction", confidence


def answer_support_relation(geometry: dict[str, Any], target_objects: list[str]) -> tuple[str, str, float]:
    objects = object_by_label(geometry)
    candidates = target_objects or list(objects.keys())
    best_label = "unknown"
    best_score = -1.0

    for label in candidates:
        obj = objects[label]
        score = 0.0
        area = numeric_value(obj, "mask_area_fraction") or 0.0
        score += min(area * 4.0, 1.0)
        if obj.get("vertical_position") in {"middle", "bottom"}:
            score += 0.5
        if obj.get("depth_order_hint") in {"middle", "near"}:
            score += 0.25
        if score > best_score:
            best_score = score
            best_label = label

    confidence = 0.45 if best_label != "unknown" else 0.0
    return best_label, "support_heuristic", confidence


def normalize_answers(values: Any) -> set[str]:
    if not isinstance(values, list):
        values = [values]
    return {str(value).strip().lower() for value in values if value is not None}


def is_correct(prediction: str, record: dict[str, Any]) -> bool:
    acceptable = normalize_answers(record.get("acceptable_answers") or record.get("answer"))
    return prediction.strip().lower() in acceptable


def base_result(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": record.get("question_id"),
        "image": record.get("image"),
        "type": record.get("type"),
        "question": record.get("question"),
        "target_objects": record.get("target_objects", []),
        "answer": record.get("answer"),
        "acceptable_answers": record.get("acceptable_answers", []),
    }


def answer_record(record: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    result = base_result(record)
    missing = record.get("missing_target_objects", [])
    if missing:
        result.update(
            {
                "prediction": "unknown",
                "source": "none",
                "confidence": 0.0,
                "correct": False,
                "error_reason": "missing_target_objects",
                "missing_target_objects": missing,
            }
        )
        return result

    question_type = record.get("type")
    target_objects = record.get("target_objects", [])
    if not isinstance(target_objects, list):
        target_objects = []
    target_objects = [str(label) for label in target_objects]

    if question_type not in SUPPORTED_TYPES:
        prediction, source, confidence = "unknown", "unsupported_question_type", 0.0
    elif question_type == "closer_farther" and len(target_objects) >= 2:
        prediction, source, confidence = answer_from_depth_relation(geometry, target_objects)
    elif question_type == "physical_size" and len(target_objects) >= 2:
        prediction, source, confidence = answer_from_area(geometry, target_objects)
    elif question_type == "support_relation":
        prediction, source, confidence = answer_support_relation(geometry, target_objects)
    else:
        prediction, source, confidence = "unknown", "insufficient_target_objects", 0.0

    result.update(
        {
            "prediction": prediction,
            "source": source,
            "confidence": confidence,
            "correct": is_correct(prediction, record),
            "error_reason": None if prediction != "unknown" else source,
            "missing_target_objects": missing,
        }
    )
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for result in results if result.get("correct"))
    answered = sum(1 for result in results if result.get("prediction") != "unknown")
    unknown = total - answered
    end_to_end_accuracy = round(correct / total, 6) if total else 0.0
    answered_accuracy = round(correct / answered, 6) if answered else 0.0
    coverage = round(answered / total, 6) if total else 0.0
    unknown_rate = round(unknown / total, 6) if total else 0.0
    by_type: dict[str, dict[str, Any]] = {}
    for question_type, group_count in collections.Counter(str(result.get("type")) for result in results).items():
        group = [result for result in results if str(result.get("type")) == question_type]
        group_correct = sum(1 for result in group if result.get("correct"))
        group_answered = sum(1 for result in group if result.get("prediction") != "unknown")
        group_unknown = group_count - group_answered
        group_end_to_end_accuracy = round(group_correct / group_count, 6) if group_count else 0.0
        by_type[question_type] = {
            "total": group_count,
            "answered": group_answered,
            "unknown": group_unknown,
            "correct": group_correct,
            "end_to_end_accuracy": group_end_to_end_accuracy,
            "answered_accuracy": round(group_correct / group_answered, 6) if group_answered else 0.0,
            "accuracy": group_end_to_end_accuracy,
            "coverage": round(group_answered / group_count, 6) if group_count else 0.0,
            "unknown_rate": round(group_unknown / group_count, 6) if group_count else 0.0,
        }
    return {
        "baseline": "geometry_rule",
        "total": total,
        "answered": answered,
        "unknown": unknown,
        "correct": correct,
        "end_to_end_accuracy": end_to_end_accuracy,
        "answered_accuracy": answered_accuracy,
        "accuracy": end_to_end_accuracy,
        "coverage": coverage,
        "unknown_rate": unknown_rate,
        "by_type": by_type,
        "error_reasons": dict(collections.Counter(str(result.get("error_reason")) for result in results if result.get("error_reason"))),
    }


def run_baseline_file(prompt_path: Path, output_path: Path, summary_path: Path, overwrite: bool) -> int:
    existing = [path for path in (output_path, summary_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Baseline output already exists: {existing[0]}")

    prompt_records = load_jsonl(prompt_path)
    results = []
    for record in prompt_records:
        geometry_path = Path(str(record.get("geometry_path", "")))
        if not geometry_path.exists():
            raise FileNotFoundError(f"Geometry file not found: {geometry_path}")
        geometry = load_json(geometry_path)
        results.append(answer_record(record, geometry))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summarize_results(results), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a geometry-only rule baseline for GeoVLM questions.")
    parser.add_argument("--prompts", type=Path, default=Path("outputs/reasoning/prompts.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reasoning/geometry_rule_baseline.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/evaluations/geometry_rule_baseline_summary.json"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing baseline outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        count = run_baseline_file(
            prompt_path=args.prompts,
            output_path=args.output,
            summary_path=args.summary,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print(f"Wrote {args.output} ({count} baseline results)")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
