"""Normalize detector labels before segmentation and geometry stages.

Usage: python scripts/06_normalize_detections.py --overwrite
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def collect_detection_files(detection_dir: Path) -> list[Path]:
    if not detection_dir.exists():
        return []
    return sorted(
        (path for path in detection_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json"),
        key=lambda path: path.name.lower(),
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_label_alias(
    image_name: str,
    label: str,
    correction_payload: dict[str, Any],
) -> tuple[str | None, str | None]:
    image_aliases = correction_payload.get("image_label_aliases", {})
    if isinstance(image_aliases, dict):
        aliases_for_image = image_aliases.get(image_name, {})
        if isinstance(aliases_for_image, dict) and label in aliases_for_image:
            return str(aliases_for_image[label]), "image_label_aliases"

    global_aliases = correction_payload.get("global_label_aliases", {})
    if isinstance(global_aliases, dict) and label in global_aliases:
        return str(global_aliases[label]), "global_label_aliases"

    return None, None


def normalize_detection_payload(
    detection_payload: dict[str, Any],
    correction_payload: dict[str, Any],
) -> dict[str, Any]:
    normalized = deepcopy(detection_payload)
    image_name = str(normalized.get("image", ""))

    objects = normalized.get("objects", [])
    if not isinstance(objects, list):
        normalized["objects"] = []
        return normalized

    for detection in objects:
        if not isinstance(detection, dict):
            continue
        label = detection.get("label")
        if not isinstance(label, str):
            continue

        alias, rule_name = get_label_alias(
            image_name=image_name,
            label=label,
            correction_payload=correction_payload,
        )
        if alias is None or rule_name is None or alias == label:
            continue

        detection["raw_label"] = label
        detection["label"] = alias
        detection["normalization"] = {
            "rule": rule_name,
            "from": label,
            "to": alias,
        }

    normalized["normalization_config"] = correction_payload.get("version", "unknown")
    return normalized


def write_normalized_payload(payload: dict[str, Any], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_detection_file(
    detection_path: Path,
    output_dir: Path,
    correction_payload: dict[str, Any],
    overwrite: bool,
) -> Path:
    detection_payload = load_json(detection_path)
    normalized_payload = normalize_detection_payload(
        detection_payload,
        correction_payload=correction_payload,
    )
    output_path = output_dir / detection_path.name
    write_normalized_payload(normalized_payload, output_path, overwrite=overwrite)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize GeoVLM detection labels using explicit rules.")
    parser.add_argument("--detection-dir", type=Path, default=Path("outputs/detections"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/detections_normalized"))
    parser.add_argument("--corrections", type=Path, default=Path("configs/detection_corrections.example.json"))
    parser.add_argument("--limit", type=int, default=None, help="Normalize only the first N detection files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing normalized detection files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detection_files = collect_detection_files(args.detection_dir)
    if args.limit is not None:
        detection_files = detection_files[: args.limit]

    if not detection_files:
        print(f"No detection JSON files found in {args.detection_dir}")
        return 1

    correction_payload = load_json(args.corrections)
    changed_objects = 0
    try:
        for detection_path in detection_files:
            output_path = normalize_detection_file(
                detection_path=detection_path,
                output_dir=args.output_dir,
                correction_payload=correction_payload,
                overwrite=args.overwrite,
            )
            payload = load_json(output_path)
            changed = sum(1 for item in payload.get("objects", []) if isinstance(item, dict) and "raw_label" in item)
            changed_objects += changed
            print(f"Wrote {output_path} ({changed} normalized objects)")
    except (FileExistsError, ValueError) as error:
        print(error)
        return 1

    print(f"Normalized {len(detection_files)} detection files.")
    print(f"Objects changed: {changed_objects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
