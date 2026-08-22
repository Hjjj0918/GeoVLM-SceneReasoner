"""Run one public SPAR inference track.

Examples:
    python scripts/23_run_public_spar_inference.py --track pure_vlm --provider mock --limit 5 --overwrite
    python scripts/23_run_public_spar_inference.py --track geovlm --provider openai_compatible --model Qwen3-VL-Flash
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.public_spar_inference import (  # noqa: E402
    TRACKS,
    MockProvider,
    OpenAICompatibleProvider,
    run_public_inference_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a public SPAR track.")
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument("--provider", choices=("mock", "openai_compatible"), default="mock")
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--prompts", type=Path, default=Path("outputs/public_spar/reasoning/prompts.jsonl"))
    parser.add_argument("--subset", type=Path, default=Path("outputs/public_spar/subsets/phase_a_30.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/public_spar/images"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--mock-response", default="unknown")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


def build_provider(args: argparse.Namespace):
    if args.provider == "mock":
        return MockProvider(model_name=args.model, response=args.mock_response)
    return OpenAICompatibleProvider(
        model_name=args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
    )


def main() -> int:
    args = parse_args()
    output_path = args.output or Path(f"outputs/public_spar/inference/{args.track}.jsonl")
    summary_path = args.summary or Path(f"outputs/public_spar/evaluations/{args.track}_summary.json")
    try:
        count = run_public_inference_file(
            prompt_path=args.prompts,
            subset_path=args.subset,
            images_dir=args.images_dir,
            output_path=output_path,
            summary_path=summary_path,
            track=args.track,
            provider=build_provider(args),
            overwrite=args.overwrite,
            limit=args.limit,
            progress_stream=sys.stdout if args.progress else None,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Wrote {output_path} ({count} inference results)")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
