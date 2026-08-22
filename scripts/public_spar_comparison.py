"""Comparison utilities for the three public SPAR inference tracks."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.public_spar_inference import TRACKS, summarize_public_results


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(record)
    return records


def _index(records: list[dict[str, Any]], track: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = record.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{track} contains a record without question_id")
        if question_id in indexed:
            raise ValueError(f"{track} contains duplicate question_id: {question_id}")
        indexed[question_id] = record
    return indexed


def _aligned_rows(indexed: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    ids = {track: set(records) for track, records in indexed.items()}
    expected = ids[TRACKS[0]]
    for track in TRACKS[1:]:
        if ids[track] != expected:
            missing = sorted(expected - ids[track])
            extra = sorted(ids[track] - expected)
            raise ValueError(f"question IDs differ for {track}: missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    for question_id in sorted(expected):
        base = indexed[TRACKS[0]][question_id]
        base_definition = _evaluation_signature(base)
        for track in TRACKS[1:]:
            if _evaluation_signature(indexed[track][question_id]) != base_definition:
                raise ValueError(
                    f"evaluation definitions differ for {question_id}: "
                    f"{TRACKS[0]}={base_definition}, {track}={_evaluation_signature(indexed[track][question_id])}"
                )
        row: dict[str, Any] = {
            "question_id": question_id,
            "image": base.get("image"),
            "task": base.get("task", base.get("type")),
            "format_type": base.get("format_type"),
            "img_type": base.get("img_type"),
        }
        for track in TRACKS:
            record = indexed[track][question_id]
            row[f"{track}_prediction"] = record.get("prediction")
            row[f"{track}_correct"] = record.get("correct") is True
            row[f"{track}_parse_status"] = record.get("parse_status")
            row[f"{track}_raw_response"] = record.get("raw_response")
        rows.append(row)
    return rows


def _evaluation_signature(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Return fields that must remain identical across aligned track outputs."""

    return (
        record.get("metric"),
        record.get("unit"),
        record.get("tolerance"),
    )


def _pairwise(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("Cannot compare tracks with different question IDs")
    both_correct = left_only = right_only = both_wrong = 0
    for question_id in left:
        left_correct = left[question_id].get("correct") is True
        right_correct = right[question_id].get("correct") is True
        if left_correct and right_correct:
            both_correct += 1
        elif left_correct:
            left_only += 1
        elif right_correct:
            right_only += 1
        else:
            both_wrong += 1
    total = len(left)
    left_name = str(next(iter(left.values())).get("track", "left")) if left else "left"
    right_name = str(next(iter(right.values())).get("track", "right")) if right else "right"
    return {
        "left_track": left_name,
        "right_track": right_name,
        "total": total,
        "left_accuracy": _ratio(both_correct + left_only, total),
        "right_accuracy": _ratio(both_correct + right_only, total),
        "delta_accuracy_right_minus_left": round(
            _ratio(both_correct + right_only, total) - _ratio(both_correct + left_only, total),
            6,
        ),
        "both_correct": both_correct,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_wrong": both_wrong,
    }


def build_public_comparison(records_by_track: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    missing = [track for track in TRACKS if track not in records_by_track]
    if missing:
        raise ValueError(f"Missing tracks: {missing}")
    indexed = {track: _index(records_by_track[track], track) for track in TRACKS}
    rows = _aligned_rows(indexed)
    disagreements = [
        row
        for row in rows
        if len({row[f"{track}_prediction"] for track in TRACKS}) > 1
        or len({row[f"{track}_correct"] for track in TRACKS}) > 1
    ]
    return {
        "tracks": {
            track: summarize_public_results(records_by_track[track], track=track, model=_model(records_by_track[track]))
            for track in TRACKS
        },
        "question_count": len(rows),
        "pairwise": {
            f"{left}_vs_{right}": _pairwise(indexed[left], indexed[right])
            for left, right in itertools.combinations(TRACKS, 2)
        },
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
    }


def _model(records: list[dict[str, Any]]) -> str:
    models = sorted({str(record.get("model")) for record in records if record.get("model") is not None})
    return models[0] if len(models) == 1 else (", ".join(models) if models else "unknown")


def write_json(payload: Mapping[str, Any], path: Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_disagreements_csv(rows: list[dict[str, Any]], path: Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id", "image", "task", "format_type", "img_type",
        *[f"{track}_{field}" for track in TRACKS for field in ("prediction", "correct", "parse_status", "raw_response")],
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

__all__ = [
    "build_public_comparison",
    "load_jsonl",
    "write_disagreements_csv",
    "write_json",
]
