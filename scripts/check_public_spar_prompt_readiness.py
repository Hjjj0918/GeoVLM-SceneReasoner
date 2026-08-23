"""Check that the canonical public SPAR prompts are ready for a real run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_public_spar_prompts import audit_prompt_records, load_jsonl as load_prompt_jsonl


def load_subset_question_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("question_ids")
    if isinstance(ids, list):
        return [str(value) for value in ids]
    questions = payload.get("questions", [])
    if isinstance(questions, list):
        extracted = [str(item.get("question_id")) for item in questions if isinstance(item, Mapping)]
        if extracted:
            return extracted
    raise ValueError(f"Subset must contain either question_ids or questions: {path}")


def check_prompt_readiness(
    subset_path: Path,
    prompts_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    subset_question_ids = load_subset_question_ids(subset_path)
    prompt_records = load_prompt_jsonl(prompts_path)
    prompt_question_ids = [str(record.get("question_id")) for record in prompt_records]
    audit = audit_prompt_records(prompt_records)

    subset_set = set(subset_question_ids)
    prompt_set = set(prompt_question_ids)
    missing_question_ids = sorted(subset_set - prompt_set)
    extra_question_ids = sorted(prompt_set - subset_set)
    prompt_audit_ready = (
        audit["leading_newline_count"] == 0
        and audit["missing_front_loaded_decision_aid_count"] == 0
    )
    ready = (
        not missing_question_ids
        and not extra_question_ids
        and len(prompt_records) == len(subset_question_ids)
        and prompt_audit_ready
    )
    report = {
        "ready": ready,
        "subset_count": len(subset_question_ids),
        "prompt_count": len(prompt_records),
        "missing_question_ids": missing_question_ids,
        "extra_question_ids": extra_question_ids,
        "prompt_audit_ready": prompt_audit_ready,
        **audit,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Check readiness of public SPAR prompts for inference.")
    parser.add_argument(
        "--subset",
        type=Path,
        default=Path("outputs/public_spar/subsets/phase_a_geometry_ready_24.json"),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("outputs/public_spar/reasoning/prompts_geometry_ready_24.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/public_spar/evaluations/prompt_readiness.json"),
    )
    args = parser.parse_args()

    try:
        report = check_prompt_readiness(args.subset, args.prompts, args.output)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"Subset questions: {report['subset_count']}")
    print(f"Prompt records: {report['prompt_count']}")
    print(f"Ready: {report['ready']}")
    print(f"Missing questions: {len(report['missing_question_ids'])}")
    print(f"Extra questions: {len(report['extra_question_ids'])}")
    print(f"Audit newline prompts: {report['leading_newline_count']}")
    print(f"Audit missing decision aids: {report['missing_front_loaded_decision_aid_count']}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
