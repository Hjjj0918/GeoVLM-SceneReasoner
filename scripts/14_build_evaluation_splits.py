"""Build evaluation splits for geometry and upstream pipeline review.

Usage: python scripts/14_build_evaluation_splits.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REVIEW_STATUS = "geometry_available_candidates require manual mask/depth review"
PIPELINE_FAILURE_REASON = "missing_target_objects"
MANUAL_REVIEW_FIELDS = ["mask_ok", "depth_ok", "question_valid", "final_split", "review_notes"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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


def automatic_split(record: dict[str, Any]) -> str:
    if record.get("error_reason") == PIPELINE_FAILURE_REASON:
        return "pipeline_failure"
    return "geometry_available_candidates"


def build_split_payload(records: list[dict[str, Any]], source: str | None = None) -> dict[str, Any]:
    candidates = [
        str(record.get("question_id", ""))
        for record in records
        if automatic_split(record) == "geometry_available_candidates"
    ]
    pipeline_failures = [
        str(record.get("question_id", ""))
        for record in records
        if automatic_split(record) == "pipeline_failure"
    ]
    payload: dict[str, Any] = {
        "version": "0.1",
        "source": source or "outputs/reasoning/geometry_rule_baseline.jsonl",
        "review_status": REVIEW_STATUS,
        "counts": {
            "total": len(records),
            "geometry_available_candidates": len(candidates),
            "pipeline_failure": len(pipeline_failures),
            "manual_review_pending": len(candidates),
        },
        "splits": {
            "geometry_available_candidates": candidates,
            "pipeline_failure": pipeline_failures,
        },
    }
    return payload


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def review_rows(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for record in records:
        split = automatic_split(record)
        review_status = REVIEW_STATUS if split == "geometry_available_candidates" else "pipeline_failure requires upstream review"
        rows.append(
            {
                "question_id": str(record.get("question_id", "")),
                "image": str(record.get("image", "")),
                "type": str(record.get("type", "")),
                "target_objects": _json_text(record.get("target_objects", [])),
                "automatic_split": split,
                "review_status": review_status,
                "prediction": str(record.get("prediction", "")),
                "correct": str(bool(record.get("correct", False))).lower(),
                "error_reason": str(record.get("error_reason") or ""),
                "missing_target_objects": _json_text(record.get("missing_target_objects", [])),
                "mask_ok": "",
                "depth_ok": "",
                "question_valid": "",
                "final_split": "",
                "review_notes": "",
            }
        )
    return rows


def write_review_csv(records: list[dict[str, Any]], review_path: Path) -> None:
    fieldnames = [
        "question_id",
        "image",
        "type",
        "target_objects",
        "automatic_split",
        "review_status",
        "prediction",
        "correct",
        "error_reason",
        "missing_target_objects",
        *MANUAL_REVIEW_FIELDS,
    ]
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows(records))


def write_split_files(
    baseline_path: Path,
    split_path: Path,
    review_path: Path,
    overwrite: bool,
) -> int:
    existing = [path for path in (split_path, review_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Evaluation split output already exists: {existing[0]}")

    records = load_jsonl(baseline_path)
    payload = build_split_payload(records, source=str(baseline_path).replace("\\", "/"))

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_review_csv(records, review_path)
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GeoVLM evaluation splits for manual review.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("outputs/reasoning/geometry_rule_baseline.jsonl"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("outputs/evaluations/evaluation_splits.json"),
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("outputs/evaluations/evaluation_review.csv"),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        count = write_split_files(
            baseline_path=args.baseline,
            split_path=args.split,
            review_path=args.review,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print(f"Wrote {args.split}")
    print(f"Wrote {args.review}")
    print(f"Read {count} baseline records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
