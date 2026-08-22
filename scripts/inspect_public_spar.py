"""Inspect SPAR-Bench rows without materializing full media arrays in outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from scripts.public_spar_common import json_safe, normalize_source_id, to_list, write_json


INSPECTION_FIELDS = (
    "id",
    "img_type",
    "format_type",
    "task",
    "source",
    "image",
    "depth",
    "pose",
    "intrinsic_color",
    "intrinsic_depth",
    "question",
    "answer",
)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else "unknown"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return len(value) == 0
    except TypeError:
        return False


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return [int(item) for item in shape]
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        result: list[int] = []
        current: Any = value
        while isinstance(current, (list, tuple)):
            result.append(len(current))
            if not current:
                break
            current = current[0]
        return result
    return None


def _describe(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(value).__name__}
    if _is_missing(value):
        result["missing"] = True
        return result

    shape = _shape(value)
    if shape is not None:
        result["shape"] = shape
    if isinstance(value, (str, bytes, bytearray)):
        result["preview"] = str(value)[:120]
        result["length"] = len(value)
        return result
    try:
        items = to_list(value)
    except Exception:
        items = []
    if items:
        result["length"] = len(items)
        first = items[0]
        if isinstance(first, (str, int, float, bool)):
            result["first_item"] = json_safe(first)
    else:
        result["length"] = 0
    size = getattr(value, "size", None)
    if isinstance(size, tuple):
        result["size"] = list(size)
    mode = getattr(value, "mode", None)
    if isinstance(mode, str):
        result["mode"] = mode
    return result


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    task_counts = Counter(_text(row.get("task")) for row in materialized)
    format_counts = Counter(_text(row.get("format_type")) for row in materialized)
    img_type_counts = Counter(_text(row.get("img_type")) for row in materialized)
    view_counts = Counter(str(len(to_list(row.get("image")))) for row in materialized)
    answer_types = Counter(type(row.get("answer")).__name__ for row in materialized)
    question_lengths = [len(_text(row.get("question"))) for row in materialized]
    missing_fields = Counter(
        field
        for row in materialized
        for field in INSPECTION_FIELDS
        if field not in row or _is_missing(row.get(field))
    )
    return {
        "dataset": "spar_bench_tiny_rgbd",
        "total_rows": len(materialized),
        "fields_seen": sorted({key for row in materialized for key in row}),
        "task_counts": dict(sorted(task_counts.items())),
        "format_type_counts": dict(sorted(format_counts.items())),
        "img_type_counts": dict(sorted(img_type_counts.items())),
        "view_count": dict(sorted(view_counts.items(), key=lambda item: int(item[0]))),
        "answer_type_counts": dict(sorted(answer_types.items())),
        "question_length": {
            "min": min(question_lengths) if question_lengths else 0,
            "max": max(question_lengths) if question_lengths else 0,
            "mean": round(sum(question_lengths) / len(question_lengths), 3)
            if question_lengths
            else 0.0,
        },
        "missing_field_counts": dict(sorted(missing_fields.items())),
    }


def build_preview(rows: Iterable[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    preview = []
    for row in list(rows)[: max(0, limit)]:
        preview.append(
            {
                "source_id": normalize_source_id(row.get("id")),
                "task": _text(row.get("task")),
                "img_type": _text(row.get("img_type")),
                "format_type": _text(row.get("format_type")),
                "source": json_safe(row.get("source")),
                "image": _describe(row.get("image")),
                "depth": _describe(row.get("depth")),
                "pose": _describe(row.get("pose")),
                "intrinsic_color": _describe(row.get("intrinsic_color")),
                "intrinsic_depth": _describe(row.get("intrinsic_depth")),
                "question": _text(row.get("question"))[:500],
                "answer": json_safe(row.get("answer")),
            }
        )
    return preview


def write_inspection_outputs(
    rows: Iterable[dict[str, Any]],
    output_dir: Path,
    preview_limit: int = 20,
    overwrite: bool = False,
) -> list[Path]:
    materialized = list(rows)
    summary_path = output_dir / "inspection_summary.json"
    distribution_path = output_dir / "task_distribution.csv"
    preview_path = output_dir / "sample_preview.json"
    paths = [summary_path, distribution_path, preview_path]
    existing = next((path for path in paths if path.exists()), None)
    if existing and not overwrite:
        raise FileExistsError(f"Output already exists: {existing}")

    write_json(summarize_rows(materialized), summary_path, overwrite=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with distribution_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dimension", "value", "count"])
        summary = summarize_rows(materialized)
        for dimension in ("task_counts", "format_type_counts", "img_type_counts"):
            for value, count in summary[dimension].items():
                writer.writerow([dimension, value, count])
    write_json(build_preview(materialized, preview_limit), preview_path, overwrite=True)
    return paths


def load_dataset_rows(
    dataset_name: str = "jasonzhango/SPAR-Bench-Tiny-RGBD",
    split: str = "test",
    streaming: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    return list(
        iter_dataset_rows(
            dataset_name=dataset_name,
            split=split,
            streaming=streaming,
            limit=limit,
        )
    )


def iter_dataset_rows(
    dataset_name: str = "jasonzhango/SPAR-Bench-Tiny-RGBD",
    split: str = "test",
    streaming: bool = False,
    limit: int | None = None,
    columns: list[str] | None = None,
):
    """Yield dataset rows without retaining the dataset in a Python list."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Install datasets with: python -m pip install datasets") from error
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    if columns is not None:
        available = set(getattr(dataset, "column_names", []) or [])
        requested = [column for column in columns if column in available]
        if requested and hasattr(dataset, "select_columns"):
            dataset = dataset.select_columns(requested)
    for index, row in enumerate(dataset):
        yield dict(row)
        if limit is not None and index + 1 >= limit:
            break
