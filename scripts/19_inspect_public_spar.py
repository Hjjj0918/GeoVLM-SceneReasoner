"""Inspect the Tiny-RGBD schema and write bounded JSON/CSV summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inspect_public_spar import load_dataset_rows, write_inspection_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SPAR-Bench-Tiny-RGBD rows.")
    parser.add_argument("--dataset", default="jasonzhango/SPAR-Bench-Tiny-RGBD")
    parser.add_argument("--split", default="test")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview-limit", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/public_spar"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = load_dataset_rows(
            dataset_name=args.dataset,
            split=args.split,
            streaming=args.streaming,
            limit=args.limit,
        )
        paths = write_inspection_outputs(
            rows, output_dir=args.output_dir, preview_limit=args.preview_limit, overwrite=args.overwrite
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(error)
        return 1
    print(f"Inspected rows: {len(rows)}")
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
