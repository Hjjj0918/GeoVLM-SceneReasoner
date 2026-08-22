"""Prepare a deterministic public SPAR subset and export its RGB views."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_public_spar import DEFAULT_SELECTION, load_rows_from_huggingface, write_phase_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SPAR-Bench-Tiny-RGBD subset.")
    parser.add_argument("--dataset", default="jasonzhango/SPAR-Bench-Tiny-RGBD")
    parser.add_argument("--split", default="test")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--phase", choices=["a", "b", "c"], required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/public_spar_selection.example.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/public_spar/subsets"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/public_spar/images"))
    parser.add_argument("--questions-output", type=Path, default=Path("data/public_spar/questions.spar.json"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = DEFAULT_SELECTION
        if args.config.exists():
            config = {**DEFAULT_SELECTION, **json.loads(args.config.read_text(encoding="utf-8"))}
        rows = load_rows_from_huggingface(args.dataset, args.split, args.streaming)
        subset_path, audit_path = write_phase_outputs(
            rows,
            args.phase,
            args.output_root,
            args.images_dir,
            config=config,
            questions_path=args.questions_output,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(error)
        return 1
    print(f"Wrote {subset_path}")
    print(f"Wrote {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
