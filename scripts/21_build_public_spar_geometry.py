"""Build oracle geometry files from a selected public SPAR subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_public_spar_geometry import build_geometry_files
from scripts.inspect_public_spar import load_dataset_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build answer-free SPAR oracle geometry.")
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--dataset", default="jasonzhango/SPAR-Bench-Tiny-RGBD")
    parser.add_argument("--split", default="test")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/public_spar/geometry_oracle"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        subset = json.loads(args.subset.read_text(encoding="utf-8"))
        question_records = subset.get("questions", [])
        rows = load_dataset_rows(args.dataset, args.split, args.streaming)
        paths = build_geometry_files(rows, question_records, args.output_dir, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(error)
        return 1
    print(f"Wrote {len(paths)} oracle geometry files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
