"""Shared, network-free helpers for the public SPAR benchmark pipeline."""

from __future__ import annotations

import datetime as _datetime
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DATASET_NAME = "spar_bench_tiny_rgbd"
SUPPORTED_METRICS = {"exact_match", "multiple_choice", "numeric_tolerance"}
SUPPORTED_IMG_TYPES = {"single_view"}
DEFAULT_NUMERIC_TOLERANCE = 0.1


def json_safe(value: Any) -> Any:
    """Convert common dataset values into JSON-serializable Python values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return json_safe(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return json_safe(item())
    return str(value)


def normalize_source_id(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text or "unknown"


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def deterministic_question_id(dataset: str, source_id: Any) -> str:
    normalized = normalize_source_id(source_id)
    prefix = "spar_tiny" if dataset == DATASET_NAME else "spar"
    return f"{prefix}_{normalized.zfill(6) if normalized.isdigit() else normalized}"


def _answer_is_numeric(answer: Any) -> bool:
    if isinstance(answer, bool) or answer is None:
        return False
    if isinstance(answer, (int, float)):
        return math.isfinite(float(answer))
    if isinstance(answer, str):
        try:
            float(answer.strip().replace(",", ""))
        except ValueError:
            return False
        return True
    return False


def classify_answer_type(row: Mapping[str, Any]) -> str:
    answer = row.get("answer")
    format_type = str(row.get("format_type", "")).strip().lower()
    if _answer_is_numeric(answer) and format_type in {"fill", "", "numeric"}:
        return "numeric"
    if format_type in {"select", "multiple_choice", "choice"}:
        return "multiple_choice"
    if isinstance(answer, (str, int, float)):
        return "exact"
    return "structured"


def _infer_metric(row: Mapping[str, Any], answer_type: str) -> str:
    format_type = str(row.get("format_type", "")).strip().lower()
    if answer_type == "numeric" or format_type in {"fill_numeric", "numeric"}:
        return "numeric_tolerance"
    if format_type in {"select", "multiple_choice", "choice"}:
        return "multiple_choice"
    if format_type in {"sentence", "generation", "long_generation"}:
        return "unsupported"
    return "exact_match"


def parse_metric_config(row: Mapping[str, Any]) -> dict[str, Any]:
    supplied = row.get("evaluation")
    supplied = dict(supplied) if isinstance(supplied, Mapping) else {}
    answer_type = classify_answer_type(row)
    metric = str(supplied.get("metric") or _infer_metric(row, answer_type)).strip()
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"unsupported evaluation metric: {metric}")

    config: dict[str, Any] = {"metric": metric}
    if metric == "numeric_tolerance":
        tolerance = supplied.get("absolute_tolerance", DEFAULT_NUMERIC_TOLERANCE)
        try:
            tolerance = float(tolerance)
        except (TypeError, ValueError) as error:
            raise ValueError("numeric absolute_tolerance must be a finite number") from error
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("numeric absolute_tolerance must be a finite number >= 0")
        config["absolute_tolerance"] = tolerance
        config["unit"] = str(supplied.get("unit", "meter"))
    if metric == "multiple_choice":
        config["acceptable_answers"] = json_safe(
            supplied.get("acceptable_answers", row.get("answer"))
        )
    return config


def utc_timestamp() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).isoformat()


def write_json(payload: Any, path: Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
