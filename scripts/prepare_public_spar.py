"""Select and convert SPAR rows into the repository-owned public schema."""

from __future__ import annotations

import json
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from scripts.public_spar_common import (
    DATASET_NAME,
    SUPPORTED_METRICS,
    classify_answer_type,
    deterministic_question_id,
    json_safe,
    normalize_source_id,
    parse_metric_config,
    to_list,
    utc_timestamp,
    write_json,
)


DEFAULT_SELECTION: dict[str, Any] = {
    "img_types": ["single_view"],
    "format_priority": ["select", "fill"],
    "task_keywords": ["depth", "distance", "spatial", "relation"],
    "excluded_modalities": ["video", "multi_view"],
    "supported_metrics": sorted(SUPPORTED_METRICS),
    "seed": 20260821,
    "per_task_limit": 20,
}
PHASE_LIMITS = {"a": 30, "b": 150, "c": None}


def _text(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _row_sort_key(row: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[Any, ...]:
    format_priority = list(config.get("format_priority", []))
    format_type = _text(row.get("format_type"))
    try:
        format_index = format_priority.index(format_type)
    except ValueError:
        format_index = len(format_priority)
    source_id = normalize_source_id(row.get("id"))
    return (format_index, _numeric_or_text(source_id), source_id)


def _numeric_or_text(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _exclusion_reasons(row: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    img_type = _text(row.get("img_type"))
    if img_type not in {_text(item) for item in config.get("img_types", [])}:
        reasons.append("img_type_not_allowed")
    if img_type in {_text(item) for item in config.get("excluded_modalities", [])}:
        reasons.append("excluded_modality")
    task = _text(row.get("task"))
    keywords = [_text(item) for item in config.get("task_keywords", [])]
    if keywords and not any(keyword in task for keyword in keywords):
        reasons.append("task_keyword_not_matched")
    try:
        metric = parse_metric_config(row)["metric"]
    except ValueError as error:
        reasons.append("unsupported_metric")
        metric = str(error)
    supported = {_text(item) for item in config.get("supported_metrics", SUPPORTED_METRICS)}
    if metric not in supported:
        reasons.append("metric_not_supported_by_selection")
    if not _text(row.get("question")):
        reasons.append("question_missing")
    if not to_list(row.get("image")):
        reasons.append("image_missing")
    return reasons


def select_rows(
    rows: list[dict[str, Any]], phase: str, config: Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = {**DEFAULT_SELECTION, **(dict(config) if config else {})}
    phase = phase.lower()
    if phase not in PHASE_LIMITS:
        raise ValueError(f"unsupported phase: {phase}; expected a, b, or c")
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        reasons = _exclusion_reasons(row, config)
        if reasons:
            excluded.append(
                {
                    "source_id": normalize_source_id(row.get("id")),
                    "reasons": reasons,
                    "task": row.get("task"),
                }
            )
        else:
            candidates.append(row)

    candidates.sort(key=lambda row: _row_sort_key(row, config))
    per_task_limit = config.get("per_task_limit")
    if phase == "c":
        per_task_limit = None
    selected: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    for row in candidates:
        task = _text(row.get("task")) or "unknown"
        if isinstance(per_task_limit, int) and task_counts[task] >= per_task_limit:
            excluded.append(
                {
                    "source_id": normalize_source_id(row.get("id")),
                    "reasons": ["per_task_limit"],
                    "task": row.get("task"),
                }
            )
            continue
        selected.append(row)
        task_counts[task] += 1
        limit = PHASE_LIMITS[phase]
        if limit is not None and len(selected) >= limit:
            break
    selected_ids = {normalize_source_id(row.get("id")) for row in selected}
    for row in candidates:
        source_id = normalize_source_id(row.get("id"))
        if source_id not in selected_ids and not any(
            item["source_id"] == source_id for item in excluded
        ):
            excluded.append(
                {
                    "source_id": source_id,
                    "reasons": ["phase_limit"],
                    "task": row.get("task"),
                }
            )
    return selected, excluded


def _image_filename(question_id: str, index: int, source: Any) -> str:
    suffix = ".jpg"
    name = getattr(source, "filename", None) or getattr(source, "name", None)
    if isinstance(name, str) and Path(name).suffix.lower() in {".png", ".jpeg", ".jpg", ".bmp"}:
        suffix = Path(name).suffix.lower()
    return f"{question_id}_view_{index:02d}{suffix}"


def convert_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source_id = normalize_source_id(row.get("id"))
    question_id = deterministic_question_id(DATASET_NAME, source_id)
    image_values = to_list(row.get("image"))
    images = [_image_filename(question_id, index, value) for index, value in enumerate(image_values)]
    metric = parse_metric_config(row)
    return {
        "question_id": question_id,
        "dataset": DATASET_NAME,
        "source_id": source_id,
        "source": json_safe(row.get("source")),
        "images": images,
        "question": str(row.get("question", "")).strip(),
        "task": str(row.get("task", "")).strip(),
        "format_type": str(row.get("format_type", "")).strip(),
        "img_type": str(row.get("img_type", "")).strip(),
        "answer": json_safe(row.get("answer")),
        "answer_type": classify_answer_type(row),
        "evaluation": metric,
        "geometry_source": "oracle_rgbd",
        "geometry_path": f"data/public_spar/geometry_oracle/{question_id}.json",
        "source_fields": {
            "depth_available": bool(to_list(row.get("depth"))),
            "pose_available": bool(to_list(row.get("pose"))),
            "intrinsic_color_available": bool(to_list(row.get("intrinsic_color"))),
            "intrinsic_depth_available": bool(to_list(row.get("intrinsic_depth"))),
        },
    }


def export_row_images(row: Mapping[str, Any], output_dir: Path) -> list[str]:
    question_id = deterministic_question_id(DATASET_NAME, row.get("id"))
    output_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for index, source in enumerate(to_list(row.get("image"))):
        name = _image_filename(question_id, index, source)
        target = output_dir / name
        if hasattr(source, "convert") and hasattr(source, "save"):
            source.convert("RGB").save(target, format="JPEG" if target.suffix == ".jpg" else None)
        elif isinstance(source, (str, Path)):
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Source image not found: {source_path}")
            from PIL import Image

            with Image.open(source_path) as image:
                image.convert("RGB").save(target, format="JPEG" if target.suffix == ".jpg" else None)
        else:
            raise ValueError(f"Unsupported image value for {question_id}: {type(source).__name__}")
        names.append(name)
    return names


def selection_audit(
    selected: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    phase: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reason_counts = Counter(
        reason for item in excluded for reason in item.get("reasons", [])
    )
    task_counts = Counter(_text(row.get("task")) or "unknown" for row in selected)
    return {
        "dataset": DATASET_NAME,
        "phase": phase,
        "created_at_utc": utc_timestamp(),
        "selection_config": json_safe(dict(config)),
        "selected_count": len(selected),
        "excluded_count": len(excluded),
        "selected_source_ids": [normalize_source_id(row.get("id")) for row in selected],
        "source_fingerprint_sha256": hashlib.sha256(
            "\n".join(normalize_source_id(row.get("id")) for row in selected).encode("utf-8")
        ).hexdigest(),
        "selected_task_counts": dict(sorted(task_counts.items())),
        "excluded_by_reason": dict(sorted(reason_counts.items())),
        "excluded": json_safe(excluded),
    }


def write_phase_outputs(
    rows: list[dict[str, Any]],
    phase: str,
    output_root: Path,
    images_dir: Path,
    config: Mapping[str, Any] | None = None,
    questions_path: Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    config = {**DEFAULT_SELECTION, **(dict(config) if config else {})}
    selected, excluded = select_rows(rows, phase=phase, config=config)
    converted = []
    for row in selected:
        record = convert_row(row)
        export_row_images(row, images_dir)
        converted.append(record)
    output_root.mkdir(parents=True, exist_ok=True)
    subset_path = output_root / f"phase_{phase}_{len(converted)}.json"
    audit_path = output_root / f"phase_{phase}_{len(converted)}_audit.json"
    write_json({"phase": phase, "questions": converted, "question_ids": [item["question_id"] for item in converted]}, subset_path, overwrite=overwrite)
    write_json(selection_audit(selected, excluded, phase, config), audit_path, overwrite=overwrite)
    unified_path = questions_path or Path("data/public_spar/questions.spar.json")
    write_json(
        {
            "version": "0.1",
            "dataset": DATASET_NAME,
            "phase": phase,
            "questions": converted,
        },
        unified_path,
        overwrite=overwrite,
    )
    return subset_path, audit_path


def load_rows_from_huggingface(dataset_name: str, split: str, streaming: bool = False) -> list[dict[str, Any]]:
    from scripts.inspect_public_spar import load_dataset_rows

    return load_dataset_rows(dataset_name=dataset_name, split=split, streaming=streaming)
