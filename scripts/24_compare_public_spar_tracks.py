"""Compare public SPAR pure-VLM, geometry-only, and GeoVLM outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.public_spar_comparison import (  # noqa: E402
    build_public_comparison,
    load_jsonl,
    write_disagreements_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the three public SPAR tracks.")
    parser.add_argument("--pure-vlm", type=Path, default=Path("outputs/public_spar/inference/pure_vlm.jsonl"))
    parser.add_argument("--geometry-only", type=Path, default=Path("outputs/public_spar/inference/geometry_only.jsonl"))
    parser.add_argument("--geovlm", type=Path, default=Path("outputs/public_spar/inference/geovlm.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/public_spar/evaluations/track_comparison.json"))
    parser.add_argument("--disagreements", type=Path, default=Path("outputs/public_spar/evaluations/track_disagreements.csv"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        comparison = build_public_comparison(
            {
                "pure_vlm": load_jsonl(args.pure_vlm),
                "geometry_only": load_jsonl(args.geometry_only),
                "geovlm": load_jsonl(args.geovlm),
            }
        )
        write_json(comparison, args.summary, overwrite=args.overwrite)
        write_disagreements_csv(comparison["disagreements"], args.disagreements, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"Compared {comparison['question_count']} questions")
    for track in ("pure_vlm", "geometry_only", "geovlm"):
        metrics = comparison["tracks"][track]
        print(f"{track}: {metrics['correct']}/{metrics['total']} accuracy={metrics['accuracy']:.6f}")
    print(f"Disagreements: {comparison['disagreement_count']}")
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.disagreements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
