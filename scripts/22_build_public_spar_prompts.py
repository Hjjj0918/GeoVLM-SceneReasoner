"""Build three public SPAR prompt tracks from converted questions and geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_public_spar_prompts import build_prompts_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build public SPAR prompts.")
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, default=Path("data/public_spar/geometry_oracle"))
    parser.add_argument("--output", type=Path, default=Path("outputs/public_spar/reasoning/prompts.jsonl"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        subset = json.loads(args.subset.read_text(encoding="utf-8"))
        questions = subset.get("questions", [])
        geometries = {
            str(question.get("question_id")): json.loads(
                (args.geometry_dir / f"{question['question_id']}.json").read_text(encoding="utf-8")
            )
            for question in questions
        }
        count = build_prompts_file(questions, geometries, args.output, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(error)
        return 1
    print(f"Wrote {args.output} ({count} prompt records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
