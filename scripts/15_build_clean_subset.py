"""Build a clean evaluation subset from manual review decisions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CLEAN_SPLIT = "clean"
CANDIDATE_SPLIT = "geometry_available_candidates"
REQUIRED_REVIEW_FIELDS = ["mask_ok", "depth_ok", "question_valid"]
YES_VALUES = {"yes", "y", "true", "1", "ok", "pass", "passed"}
REJECT_VALUES = {"reject", "rejected", "exclude", "excluded", "bad", "no"}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_yes(value: str | None) -> bool:
    return str(value or "").strip().lower() in YES_VALUES


def is_clean_review(row: dict[str, str]) -> bool:
    if row.get("automatic_split") != CANDIDATE_SPLIT:
        return False
    final_split = str(row.get("final_split", "")).strip().lower()
    if final_split in REJECT_VALUES:
        return False
    if final_split and final_split not in {CLEAN_SPLIT, "clean_subset"}:
        return False
    return all(is_yes(row.get(field)) for field in REQUIRED_REVIEW_FIELDS)


def build_clean_subset_payload(rows: list[dict[str, str]], source: str | None = None) -> dict[str, Any]:
    clean_rows = [row for row in rows if is_clean_review(row)]
    question_ids = [str(row.get("question_id", "")) for row in clean_rows if row.get("question_id")]
    return {
        "version": "0.1",
        "source": source or "outputs/evaluations/evaluation_review.csv",
        "criteria": {
            "automatic_split": CANDIDATE_SPLIT,
            "required_manual_fields": REQUIRED_REVIEW_FIELDS,
            "accepted_values": sorted(YES_VALUES),
        },
        "counts": {
            "review_rows": len(rows),
            "clean_subset": len(question_ids),
            "rejected_or_pending": len(rows) - len(question_ids),
        },
        "question_ids": question_ids,
        "records": clean_rows,
    }


def filter_questions_payload(questions_payload: dict[str, Any], question_ids: list[str]) -> dict[str, Any]:
    question_id_set = set(question_ids)
    questions = questions_payload.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError("questions file must contain a list field named 'questions'")
    filtered = [
        question
        for question in questions
        if isinstance(question, dict) and str(question.get("question_id", "")) in question_id_set
    ]
    output = dict(questions_payload)
    output["description"] = "Clean subset questions generated from manual review decisions."
    output["source_question_count"] = len(questions)
    output["clean_question_count"] = len(filtered)
    output["questions"] = filtered
    return output


def write_clean_subset_files(
    review_path: Path,
    questions_path: Path,
    output_path: Path,
    questions_output_path: Path,
    overwrite: bool,
) -> int:
    existing = [path for path in (output_path, questions_output_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Clean subset output already exists: {existing[0]}")

    rows = load_csv_rows(review_path)
    payload = build_clean_subset_payload(rows, source=str(review_path).replace("\\", "/"))
    questions_payload = filter_questions_payload(load_json(questions_path), payload["question_ids"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    questions_output_path.parent.mkdir(parents=True, exist_ok=True)
    questions_output_path.write_text(
        json.dumps(questions_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(payload["question_ids"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean GeoVLM evaluation subset from manual review CSV.")
    parser.add_argument("--review", type=Path, default=Path("outputs/evaluations/evaluation_review.csv"))
    parser.add_argument("--questions", type=Path, default=Path("data/questions.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluations/clean_subset.json"))
    parser.add_argument(
        "--questions-output",
        type=Path,
        default=Path("outputs/evaluations/clean_subset_questions.json"),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing clean subset outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        count = write_clean_subset_files(
            review_path=args.review,
            questions_path=args.questions,
            output_path=args.output,
            questions_output_path=args.questions_output,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print(f"Wrote {args.output}")
    print(f"Wrote {args.questions_output}")
    print(f"Clean subset questions: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
