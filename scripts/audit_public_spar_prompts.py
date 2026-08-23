"""Audit generated public SPAR prompts for front-loaded decision aids."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


FRONT_LOADED_TASKS = {"distance_infer_center_oo", "obj_spatial_relation_oo"}


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


def _prompt_for(record: Mapping[str, Any], track: str = "geovlm") -> str | None:
    prompts = record.get("prompts")
    if not isinstance(prompts, Mapping):
        return None
    prompt = prompts.get(track)
    return prompt if isinstance(prompt, str) else None


def audit_prompt_records(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    leading_newline_question_ids: list[str] = []
    missing_front_loaded_decision_aid_question_ids: list[str] = []
    front_loaded_decision_aid_question_ids: list[str] = []
    rows: list[dict[str, Any]] = []

    for record in records:
        question_id = str(record.get("question_id"))
        task = str(record.get("task", "")).strip().lower()
        prompt = _prompt_for(record, "geovlm")
        starts_with_newline = bool(prompt and prompt.startswith("\n"))
        starts_with_decision_aid = bool(prompt and prompt.startswith("Decision aid:"))
        requires_front_load = task in FRONT_LOADED_TASKS

        if starts_with_newline:
            leading_newline_question_ids.append(question_id)
        if requires_front_load:
            if starts_with_decision_aid:
                front_loaded_decision_aid_question_ids.append(question_id)
            else:
                missing_front_loaded_decision_aid_question_ids.append(question_id)

        rows.append(
            {
                "question_id": question_id,
                "task": task,
                "starts_with_newline": starts_with_newline,
                "starts_with_decision_aid": starts_with_decision_aid,
                "requires_front_loaded_decision_aid": requires_front_load,
                "prompt_length": len(prompt) if prompt is not None else None,
            }
        )

    return {
        "record_count": len(records),
        "leading_newline_count": len(leading_newline_question_ids),
        "leading_newline_question_ids": sorted(leading_newline_question_ids),
        "front_loaded_decision_aid_count": len(front_loaded_decision_aid_question_ids),
        "front_loaded_decision_aid_question_ids": sorted(front_loaded_decision_aid_question_ids),
        "missing_front_loaded_decision_aid_count": len(missing_front_loaded_decision_aid_question_ids),
        "missing_front_loaded_decision_aid_question_ids": sorted(missing_front_loaded_decision_aid_question_ids),
        "questions": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit public SPAR prompt structure.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/public_spar/reasoning/prompts_geometry_ready_24.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/public_spar/evaluations/prompts_audit.json"),
    )
    args = parser.parse_args()

    try:
        records = load_jsonl(args.input)
        report = audit_prompt_records(records)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {report['record_count']} prompt records")
    print(f"Leading newline prompts: {report['leading_newline_count']}")
    print(f"Front-loaded decision aids: {report['front_loaded_decision_aid_count']}")
    print(f"Missing front-loaded decision aids: {report['missing_front_loaded_decision_aid_count']}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
