"""Run Pure VLM, Geometry-only LLM, or GeoVLM inference on clean questions.

Usage: python scripts/16_run_vlm_inference.py --track pure_vlm --provider mock --limit 3 --overwrite

If you use the OpenAI-compatible provider, you must set the environment variable specified by --api-key-env (default: OPENAI_API_KEY) to your API key.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


TRACK_PROMPTS = {
    "pure_vlm": "pure_vlm_prompt",
    "geometry_only": "geometry_llm_prompt",
    "geovlm": "geovlm_prompt",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_clean_question_ids(path: Path) -> set[str]:
    payload = load_json(path)
    question_ids = payload.get("question_ids", [])
    if not isinstance(question_ids, list):
        raise ValueError(f"Clean subset must contain a question_ids list: {path}")
    return {str(question_id) for question_id in question_ids}


def select_prompt(record: dict[str, Any], track: str) -> str:
    if track not in TRACK_PROMPTS:
        raise ValueError(f"Unsupported track: {track}")
    prompt_field = TRACK_PROMPTS[track]
    prompt = record.get(prompt_field)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt record is missing {prompt_field}: {record.get('question_id')}")
    return prompt


def normalize_prediction(raw_response: str, target_objects: list[str]) -> str:
    text = str(raw_response or "").strip().lower()
    if not text or text == "unknown":
        return "unknown"

    matches: list[tuple[int, str]] = []
    for label in target_objects:
        label_text = str(label).strip().lower()
        if not label_text:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(label_text)}(?!\w)")
        match = pattern.search(text)
        if match:
            matches.append((match.start(), label_text))

    if not matches:
        return "unknown"
    return min(matches, key=lambda item: (item[0], -len(item[1])))[1]


def normalize_answers(values: Any) -> set[str]:
    if not isinstance(values, list):
        values = [values]
    return {str(value).strip().lower() for value in values if value is not None}


def is_correct(prediction: str, record: dict[str, Any]) -> bool:
    acceptable = normalize_answers(record.get("acceptable_answers") or record.get("answer"))
    return prediction.strip().lower() in acceptable


def image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_message_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Provider response has no choices[0].message.content: {payload}") from error

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "\n".join(text_parts)
    return str(content)


class MockProvider:
    """Return a fixed answer without network access for smoke tests."""

    def __init__(self, model_name: str, response: str):
        self.model_name = model_name
        self.response = response

    def answer(self, prompt: str, image_path: Path | None = None) -> str:
        return self.response


class OpenAICompatibleProvider:
    """Call an OpenAI-compatible chat-completions endpoint over HTTP."""

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key_env: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float = 120.0,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set")
        self.model_name = model_name
        self.endpoint = (
            api_base.rstrip("/")
            if api_base.rstrip("/").endswith("/chat/completions")
            else api_base.rstrip("/") + "/chat/completions"
        )
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def answer(self, prompt: str, image_path: Path | None = None) -> str:
        if image_path is None:
            content: Any = prompt
        else:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
            ]

        request_payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM provider HTTP {error.code}: {body}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"VLM provider connection failed: {error}") from error
        return extract_message_content(payload)


def summarize_results(results: list[dict[str, Any]], track: str, model: str) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for result in results if result.get("correct"))
    answered = sum(1 for result in results if result.get("prediction") != "unknown")
    unknown = total - answered
    by_type: dict[str, dict[str, Any]] = {}
    for question_type, group_count in collections.Counter(str(result.get("type")) for result in results).items():
        group = [result for result in results if str(result.get("type")) == question_type]
        group_correct = sum(1 for result in group if result.get("correct"))
        group_answered = sum(1 for result in group if result.get("prediction") != "unknown")
        by_type[question_type] = {
            "total": group_count,
            "answered": group_answered,
            "unknown": group_count - group_answered,
            "correct": group_correct,
            "accuracy": round(group_correct / group_count, 6) if group_count else 0.0,
            "answered_accuracy": round(group_correct / group_answered, 6) if group_answered else 0.0,
            "coverage": round(group_answered / group_count, 6) if group_count else 0.0,
        }

    errors = collections.Counter(
        str(result["error"]) for result in results if result.get("error")
    )
    return {
        "track": track,
        "model": model,
        "total": total,
        "answered": answered,
        "unknown": unknown,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "answered_accuracy": round(correct / answered, 6) if answered else 0.0,
        "coverage": round(answered / total, 6) if total else 0.0,
        "unknown_rate": round(unknown / total, 6) if total else 0.0,
        "by_type": by_type,
        "errors": dict(errors),
    }


def run_inference_file(
    prompt_path: Path,
    clean_subset_path: Path,
    images_dir: Path,
    output_path: Path,
    summary_path: Path,
    track: str,
    provider: Any,
    overwrite: bool,
    limit: int | None = None,
    progress_stream: Any | None = None,
) -> int:
    existing = [path for path in (output_path, summary_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Inference output already exists: {existing[0]}")
    if track not in TRACK_PROMPTS:
        raise ValueError(f"Unsupported track: {track}")

    clean_question_ids = load_clean_question_ids(clean_subset_path)
    prompt_records = [
        record
        for record in load_jsonl(prompt_path)
        if str(record.get("question_id", "")) in clean_question_ids
    ]
    prompt_records.sort(key=lambda record: str(record.get("question_id", "")))
    if limit is not None:
        prompt_records = prompt_records[:limit]

    results = []
    for record in prompt_records:
        prompt = select_prompt(record, track)
        image_path = None
        if track in {"pure_vlm", "geovlm"}:
            image_name = record.get("image")
            if not isinstance(image_name, str):
                raise ValueError(f"Prompt record is missing image: {record.get('question_id')}")
            image_path = images_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        started = time.perf_counter()
        raw_response = ""
        error = None
        error_type = None
        try:
            raw_response = provider.answer(prompt=prompt, image_path=image_path)
            target_objects = record.get("target_objects", [])
            if not isinstance(target_objects, list):
                target_objects = []
            prediction = normalize_prediction(raw_response, [str(item) for item in target_objects])
            correct = is_correct(prediction, record)
        except Exception as exception:
            prediction = "unknown"
            correct = False
            error_type = type(exception).__name__
            error = f"{type(exception).__name__}: {exception}"

        results.append(
            {
                "question_id": record.get("question_id"),
                "image": record.get("image"),
                "type": record.get("type"),
                "track": track,
                "model": getattr(provider, "model_name", "unknown"),
                "prediction": prediction,
                "raw_response": raw_response,
                "correct": correct,
                "error": error,
                "latency_seconds": round(time.perf_counter() - started, 6),
            }
        )
        if progress_stream is not None:
            progress = (
                f"[{len(results)}/{len(prompt_records)}] "
                f"{record.get('question_id')} {record.get('type')} -> {prediction} "
                f"correct={correct} latency={results[-1]['latency_seconds']:.2f}s"
            )
            if error_type:
                progress += f" error={error_type}"
            print(progress, file=progress_stream, flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            summarize_results(
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


def build_provider(args: argparse.Namespace) -> Any:
    if args.provider == "mock":
        return MockProvider(model_name=args.model, response=args.mock_response)
    return OpenAICompatibleProvider(
        model_name=args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM inference on the GeoVLM clean subset.")
    parser.add_argument("--track", choices=sorted(TRACK_PROMPTS), required=True)
    parser.add_argument("--provider", choices=["mock", "openai_compatible"], default="mock")
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--prompts", type=Path, default=Path("outputs/reasoning/prompts.jsonl"))
    parser.add_argument("--clean-subset", type=Path, default=Path("outputs/evaluations/clean_subset.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--mock-response", default="unknown")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output or Path(f"outputs/inference/{args.track}.jsonl")
    summary_path = args.summary or Path(f"outputs/evaluations/{args.track}_summary.json")
    try:
        count = run_inference_file(
            prompt_path=args.prompts,
            clean_subset_path=args.clean_subset,
            images_dir=args.images_dir,
            output_path=output_path,
            summary_path=summary_path,
            track=args.track,
            provider=build_provider(args),
            overwrite=args.overwrite,
            limit=args.limit,
            progress_stream=sys.stdout if args.progress else None,
        )
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(error)
        return 1

    print(f"Wrote {output_path} ({count} inference results)")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
