"""Compare completed Pure VLM, geometry-only, and GeoVLM result files.

Usage:
    python scripts/17_compare_track_results.py --overwrite
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
from pathlib import Path
from typing import Any


TRACKS = ("pure_vlm", "geometry_only", "geovlm")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load result records from a JSONL file."""
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
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(record)
    return records


def index_results(records: list[dict[str, Any]], track: str) -> dict[str, dict[str, Any]]:
    """Index records by question ID and reject duplicates or missing IDs."""
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = record.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{track} contains a record without question_id")
        if question_id in indexed:
            raise ValueError(f"{track} contains duplicate question_id: {question_id}")
        indexed[question_id] = record
    return indexed


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute overall and per-question-type metrics from inference records."""
    total = len(records)
    correct = sum(record.get("correct") is True for record in records)
    answered = sum(
        str(record.get("prediction", "")).strip().lower() != "unknown"
        for record in records
    )
    by_type: dict[str, dict[str, Any]] = {}
    groups = collections.defaultdict(list)
    for record in records:
        groups[str(record.get("type", "unknown"))].append(record)

    for question_type, group in sorted(groups.items()):
        group_total = len(group)
        group_correct = sum(record.get("correct") is True for record in group)
        group_answered = sum(
            str(record.get("prediction", "")).strip().lower() != "unknown"
            for record in group
        )
        by_type[question_type] = {
            "total": group_total,
            "answered": group_answered,
            "unknown": group_total - group_answered,
            "correct": group_correct,
            "accuracy": ratio(group_correct, group_total),
            "answered_accuracy": ratio(group_correct, group_answered),
            "coverage": ratio(group_answered, group_total),
        }

    errors = collections.Counter(
        str(record["error"]) for record in records if record.get("error")
    )
    models = sorted(
        {
            str(record.get("model"))
            for record in records
            if record.get("model") is not None
        }
    )
    return {
        "model": models[0] if len(models) == 1 else models,
        "total": total,
        "answered": answered,
        "unknown": total - answered,
        "correct": correct,
        "accuracy": ratio(correct, total),
        "answered_accuracy": ratio(correct, answered),
        "coverage": ratio(answered, total),
        "unknown_rate": ratio(total - answered, total),
        "by_type": by_type,
        "errors": dict(errors),
    }


def build_question_rows(
    indexed_results: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build aligned per-question rows after validating identical question IDs."""
    track_ids = {
        track: set(records.keys()) for track, records in indexed_results.items()
    }
    expected_ids = track_ids[TRACKS[0]]
    for track in TRACKS[1:]:
        if track_ids[track] != expected_ids:
            missing = sorted(expected_ids - track_ids[track])
            extra = sorted(track_ids[track] - expected_ids)
            raise ValueError(
                f"Question IDs differ for {track}: missing={missing}, extra={extra}"
            )

    rows = []
    for question_id in sorted(expected_ids):
        row: dict[str, Any] = {
            "question_id": question_id,
            "image": indexed_results[TRACKS[0]][question_id].get("image"),
            "type": indexed_results[TRACKS[0]][question_id].get("type"),
        }
        for track in TRACKS:
            record = indexed_results[track][question_id]
            row[f"{track}_prediction"] = record.get("prediction")
            row[f"{track}_correct"] = record.get("correct") is True
            row[f"{track}_raw_response"] = record.get("raw_response")
        rows.append(row)
    return rows


def compare_pair(
    left_records: dict[str, dict[str, Any]],
    right_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Calculate paired correctness counts for two aligned tracks."""
    if set(left_records) != set(right_records):
        raise ValueError("Cannot compare tracks with different question IDs")

    both_correct = 0
    left_only_correct = 0
    right_only_correct = 0
    both_wrong = 0
    for question_id in left_records:
        left_correct = left_records[question_id].get("correct") is True
        right_correct = right_records[question_id].get("correct") is True
        if left_correct and right_correct:
            both_correct += 1
        elif left_correct:
            left_only_correct += 1
        elif right_correct:
            right_only_correct += 1
        else:
            both_wrong += 1

    left_name = str(left_records[next(iter(left_records))].get("track", "left"))
    right_name = str(right_records[next(iter(right_records))].get("track", "right"))
    total = len(left_records)
    left_correct_total = both_correct + left_only_correct
    right_correct_total = both_correct + right_only_correct
    return {
        "left_track": left_name,
        "right_track": right_name,
        "total": total,
        "left_accuracy": ratio(left_correct_total, total),
        "right_accuracy": ratio(right_correct_total, total),
        "delta_accuracy_right_minus_left": round(
            ratio(right_correct_total, total) - ratio(left_correct_total, total),
            6,
        ),
        "both_correct": both_correct,
        "left_only_correct": left_only_correct,
        "right_only_correct": right_only_correct,
        "both_wrong": both_wrong,
    }


def build_comparison(
    records_by_track: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build the complete comparison payload for the three required tracks."""
    missing_tracks = [track for track in TRACKS if track not in records_by_track]
    if missing_tracks:
        raise ValueError(f"Missing tracks: {missing_tracks}")

    indexed = {
        track: index_results(records_by_track[track], track)
        for track in TRACKS
    }
    question_rows = build_question_rows(indexed)
    disagreements = [
        row
        for row in question_rows
        if len(
            {
                row[f"{track}_prediction"]
                for track in TRACKS
            }
        )
        > 1
        or len(
            {
                row[f"{track}_correct"]
                for track in TRACKS
            }
        )
        > 1
    ]

    pairwise = {}
    for left_track, right_track in itertools.combinations(TRACKS, 2):
        pairwise[f"{left_track}_vs_{right_track}"] = compare_pair(
            indexed[left_track],
            indexed[right_track],
        )

    return {
        "tracks": {
            track: summarize_results(records_by_track[track])
            for track in TRACKS
        },
        "question_count": len(question_rows),
        "pairwise": pairwise,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
    }


def write_json(payload: dict[str, Any], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_disagreements_csv(rows: list[dict[str, Any]], path: Path, overwrite: bool) -> None:
    """Write aligned rows with differing predictions or correctness."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "image",
        "type",
        "pure_vlm_prediction",
        "pure_vlm_correct",
        "pure_vlm_raw_response",
        "geometry_only_prediction",
        "geometry_only_correct",
        "geometry_only_raw_response",
        "geovlm_prediction",
        "geovlm_correct",
        "geovlm_raw_response",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the three GeoVLM inference tracks."
    )
    parser.add_argument(
        "--pure-vlm",
        type=Path,
        default=Path("outputs/inference/pure_vlm.jsonl"),
    )
    parser.add_argument(
        "--geometry-only",
        type=Path,
        default=Path("outputs/inference/geometry_only.jsonl"),
    )
    parser.add_argument(
        "--geovlm",
        type=Path,
        default=Path("outputs/inference/geovlm.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/evaluations/track_comparison.json"),
    )
    parser.add_argument(
        "--disagreements",
        type=Path,
        default=Path("outputs/evaluations/track_disagreements.csv"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        comparison = build_comparison(
            {
                "pure_vlm": load_jsonl(args.pure_vlm),
                "geometry_only": load_jsonl(args.geometry_only),
                "geovlm": load_jsonl(args.geovlm),
            }
        )
        write_json(comparison, args.summary, args.overwrite)
        write_disagreements_csv(
            comparison["disagreements"],
            args.disagreements,
            args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print(f"Compared {comparison['question_count']} questions")
    for track in TRACKS:
        metrics = comparison["tracks"][track]
        print(
            f"{track}: {metrics['correct']}/{metrics['total']} "
            f"accuracy={metrics['accuracy']:.6f}"
        )
    print(f"Disagreements: {comparison['disagreement_count']}")
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.disagreements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
