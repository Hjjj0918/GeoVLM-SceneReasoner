"""Inference and metric-aware evaluation for the public SPAR subset.

The prompt JSONL intentionally keeps the model-facing prompt under ``prompts``
and stores the gold answer separately for local evaluation.  This module never
adds that gold answer to a provider request.
"""

from __future__ import annotations

import collections
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping

from scripts.vlm_inference_common import MockProvider, OpenAICompatibleProvider


TRACKS = ("pure_vlm", "geometry_only", "geovlm")
IMAGE_TRACKS = {"pure_vlm", "geovlm"}
_MULTIPLE_CHOICE_RE = re.compile(r"(?i)(?:\boption\s*)?\b([A-D])\b")
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _text(value: Any) -> str:
    return str(value).strip()


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _metric_config(record_or_config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = record_or_config.get("evaluation")
    if isinstance(evaluation, Mapping):
        return dict(evaluation)
    return dict(record_or_config)


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        return "unknown"
    rendered = f"{value:.12f}".rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def _normalize_exact(value: Any) -> str:
    return " ".join(_text(value).casefold().split())


def normalize_public_prediction(raw: Any, evaluation_config: Mapping[str, Any]) -> str:
    """Normalize a provider response according to a public SPAR metric.

    Unknown/empty responses are represented uniformly as ``unknown`` so that
    coverage and parse-failure rates are measured separately from accuracy.
    """

    text = _text(raw)
    if not text or text.casefold() in {"unknown", "n/a", "na", "none"}:
        return "unknown"
    config = _metric_config(evaluation_config)
    metric = _text(config.get("metric", "exact_match")).casefold()

    if metric == "multiple_choice":
        # Prefer an explicit ``option B`` phrase, then a label on its own.
        option_match = re.search(r"(?i)\boption\s*([A-D])\b", text)
        candidates = [option_match.group(1).upper()] if option_match else []
        candidates.extend(match.group(1).upper() for match in _MULTIPLE_CHOICE_RE.finditer(text))
        for candidate in candidates:
            return candidate
        return "unknown"

    if metric == "numeric_tolerance":
        for match in _NUMBER_RE.finditer(text.replace("−", "-")):
            token = match.group(0).replace(",", "")
            try:
                value = float(token)
            except ValueError:
                continue
            return _format_number(value)
        return "unknown"

    # Exact-match prompts request a short phrase.  Remove only surrounding
    # punctuation and normalize whitespace; evaluator handles case folding.
    normalized = text.strip().strip("`\"'.,:; ")
    return normalized if normalized else "unknown"


def evaluate_prediction(prediction: Any, record: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a normalized prediction and return metric diagnostics."""

    config = _metric_config(record)
    metric = _text(config.get("metric", "exact_match")).casefold()
    normalized_prediction = _text(prediction) or "unknown"
    answer = record.get("answer")
    if answer is None:
        answer = config.get("acceptable_answers")

    if metric == "numeric_tolerance":
        try:
            predicted_value = float(normalized_prediction.replace(",", ""))
            gold_value = float(_text(answer).replace(",", ""))
            tolerance = float(config.get("absolute_tolerance", 0.1))
            correct = (
                normalized_prediction.casefold() != "unknown"
                and math.isfinite(predicted_value)
                and math.isfinite(gold_value)
                and abs(predicted_value - gold_value) <= tolerance + 1e-12
            )
            return {
                "correct": correct,
                "metric": metric,
                "gold": _format_number(gold_value),
                "absolute_error": round(abs(predicted_value - gold_value), 12)
                if math.isfinite(predicted_value) and math.isfinite(gold_value)
                else None,
                "tolerance": tolerance,
            }
        except (TypeError, ValueError):
            return {"correct": False, "metric": metric, "gold": _text(answer), "tolerance": config.get("absolute_tolerance", 0.1)}

    acceptable = config.get("acceptable_answers", answer)
    acceptable_values = {_normalize_exact(item) for item in _values(acceptable) if item is not None}
    normalized = _normalize_exact(normalized_prediction)
    return {
        "correct": normalized_prediction.casefold() != "unknown" and normalized in acceptable_values,
        "metric": metric,
        "gold": list(sorted(acceptable_values)),
    }


def _group_summary(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        groups[_text(record.get(field)) or "unknown"].append(record)
    return {
        key: _summarize_group(group)
        for key, group in sorted(groups.items())
    }


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    answered = sum(_text(item.get("prediction")).casefold() != "unknown" for item in records)
    correct = sum(item.get("correct") is True for item in records)
    parse_failures = sum(item.get("parse_status") == "parse_failure" for item in records)
    return {
        "total": total,
        "answered": answered,
        "unknown": total - answered,
        "correct": correct,
        "accuracy": _ratio(correct, total),
        "answered_accuracy": _ratio(correct, answered),
        "coverage": _ratio(answered, total),
        "parse_failures": parse_failures,
    }


def summarize_public_results(records: list[dict[str, Any]], track: str, model: str) -> dict[str, Any]:
    """Summarize public results overall and by task/format/image type."""

    summary = _summarize_group(records)
    summary.update(
        {
            "track": track,
            "model": model,
            "by_task": _group_summary(records, "task"),
            "by_format_type": _group_summary(records, "format_type"),
            "by_img_type": _group_summary(records, "img_type"),
            "errors": dict(
                collections.Counter(
                    _text(record.get("error"))
                    for record in records
                    if record.get("error")
                )
            ),
        }
    )
    return summary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(value)
    return records


def _load_subset_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("question_ids", [])
    if not isinstance(ids, list):
        raise ValueError(f"Subset must contain a question_ids list: {path}")
    return {str(value) for value in ids}


def _prompt_for(record: Mapping[str, Any], track: str) -> str:
    if track not in TRACKS:
        raise ValueError(f"Unsupported track: {track}")
    prompts = record.get("prompts")
    prompt = prompts.get(track) if isinstance(prompts, Mapping) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt record is missing prompts.{track}: {record.get('question_id')}")
    return prompt


def _image_for(record: Mapping[str, Any], images_dir: Path, track: str) -> Path | None:
    if track not in IMAGE_TRACKS:
        return None
    images = record.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], str):
        raise ValueError(f"Prompt record is missing an image for {record.get('question_id')}")
    image_path = images_dir / images[0]
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path


def run_public_inference_file(
    prompt_path: Path,
    images_dir: Path,
    output_path: Path,
    summary_path: Path,
    track: str,
    provider: Any,
    overwrite: bool,
    subset_path: Path | None = None,
    limit: int | None = None,
    progress_stream: Any | None = None,
) -> int:
    if track not in TRACKS:
        raise ValueError(f"Unsupported track: {track}")
    existing = next((path for path in (output_path, summary_path) if path.exists()), None)
    if existing is not None and not overwrite:
        raise FileExistsError(f"Inference output already exists: {existing}")

    allowed_ids = _load_subset_ids(subset_path)
    records = [
        record
        for record in load_jsonl(prompt_path)
        if allowed_ids is None or str(record.get("question_id")) in allowed_ids
    ]
    records.sort(key=lambda record: str(record.get("question_id", "")))
    if limit is not None:
        records = records[:limit]

    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        question_id = record.get("question_id")
        raw_response = ""
        error = None
        started = time.perf_counter()
        try:
            prompt = _prompt_for(record, track)
            image_path = _image_for(record, images_dir, track)
            raw_response = str(provider.answer(prompt=prompt, image_path=image_path) or "")
            prediction = normalize_public_prediction(raw_response, record.get("evaluation", {}))
            parse_status = "parse_failure" if prediction == "unknown" else "parsed"
            evaluation = evaluate_prediction(prediction, record)
        except Exception as exception:
            image_path = None
            prediction = "unknown"
            parse_status = "parse_failure"
            evaluation = evaluate_prediction(prediction, record)
            error = f"{type(exception).__name__}: {exception}"

        result = {
            "question_id": question_id,
            "images": record.get("images", []),
            "image": record.get("images", [None])[0] if record.get("images") else None,
            "task": record.get("task"),
            "type": record.get("task"),
            "format_type": record.get("format_type"),
            "img_type": record.get("img_type"),
            "track": track,
            "model": getattr(provider, "model_name", "unknown"),
            "prediction": prediction,
            "raw_response": raw_response,
            "parse_status": parse_status,
            "correct": evaluation["correct"],
            "metric": evaluation.get("metric"),
            "unit": (record.get("evaluation", {}).get("unit")
                     if isinstance(record.get("evaluation"), Mapping)
                     else None),
            "gold": evaluation.get("gold"),
            "error": error,
            "latency_seconds": round(time.perf_counter() - started, 6),
        }
        if "absolute_error" in evaluation:
            result["absolute_error"] = evaluation["absolute_error"]
            result["tolerance"] = evaluation.get("tolerance")
        results.append(result)
        if progress_stream is not None:
            message = (
                f"[{index}/{len(records)}] {question_id} {record.get('task')} -> "
                f"{prediction} correct={result['correct']}"
            )
            if error:
                message += f" error={error.split(':', 1)[0]}"
            print(message, file=progress_stream, flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            summarize_public_results(
                results,
                track=track,
                model=getattr(provider, "model_name", "unknown"),
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(results)
