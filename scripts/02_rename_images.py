"""Rename scene images into stable GeoVLM view-sequence filenames.

Usage: python scripts/02_rename_images.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class RenameOperation(NamedTuple):
    source: Path
    destination: Path


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        (path for path in image_dir.iterdir() if is_image_file(path)),
        key=lambda path: path.name.lower(),
    )


def build_rename_plan(files: list[Path], scene_id: str) -> list[RenameOperation]:
    plan: list[RenameOperation] = []
    for index, source in enumerate(files):
        destination = source.with_name(f"{scene_id}_view_{index:02d}{source.suffix}")
        plan.append(RenameOperation(source=source, destination=destination))
    return plan


def validate_rename_plan(plan: list[RenameOperation]) -> list[str]:
    errors: list[str] = []
    source_paths = {operation.source.resolve() for operation in plan}
    destination_paths: set[Path] = set()

    for operation in plan:
        if not operation.source.exists():
            errors.append(f"source does not exist: {operation.source.name}")

        resolved_destination = operation.destination.resolve()
        if resolved_destination in destination_paths:
            errors.append(f"duplicate destination: {operation.destination.name}")
        destination_paths.add(resolved_destination)

        is_same_file = operation.source.resolve() == resolved_destination
        is_planned_source = resolved_destination in source_paths
        if operation.destination.exists() and not is_same_file and not is_planned_source:
            errors.append(f"destination already exists: {operation.destination.name}")

    return errors


def write_manifest(plan: list[RenameOperation], manifest_path: Path, scene_id: str) -> None:
    payload = {
        "scene_id": scene_id,
        "renamed_count": len(plan),
        "files": [
            {
                "source": operation.source.name,
                "destination": operation.destination.name,
            }
            for operation in plan
        ],
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def apply_rename_plan(plan: list[RenameOperation], manifest_path: Path) -> None:
    if not plan:
        write_manifest(plan, manifest_path, scene_id="scene_0001")
        return

    scene_id = plan[0].destination.name.split("_view_", maxsplit=1)[0]
    active_operations = [operation for operation in plan if operation.source.resolve() != operation.destination.resolve()]
    temp_operations: list[RenameOperation] = []

    for index, operation in enumerate(active_operations):
        temp_path = operation.source.with_name(f".rename_tmp_{index:04d}_{operation.source.name}")
        operation.source.rename(temp_path)
        temp_operations.append(RenameOperation(source=temp_path, destination=operation.destination))

    for operation in temp_operations:
        operation.source.rename(operation.destination)

    write_manifest(plan, manifest_path, scene_id=scene_id)


def print_rename_plan(plan: list[RenameOperation]) -> None:
    if not plan:
        print("No image files found.")
        return

    for operation in plan:
        marker = "unchanged" if operation.source.resolve() == operation.destination.resolve() else "rename"
        print(f"{marker}: {operation.source.name} -> {operation.destination.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename GeoVLM scene images in sequence.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--scene-id", default="scene_0001")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="Apply renames. Without this flag, only print a dry run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = collect_image_files(args.image_dir)
    plan = build_rename_plan(files, scene_id=args.scene_id)
    manifest_path = args.manifest or args.image_dir / "rename_manifest.json"

    errors = validate_rename_plan(plan)
    if errors:
        print("Rename plan failed validation:")
        for error in errors:
            print(f"- {error}")
        return 1

    if not args.apply:
        print("Dry run. No files were renamed.")
        print_rename_plan(plan)
        print("\nRun again with --apply to rename files and write the manifest.")
        return 0

    apply_rename_plan(plan, manifest_path)
    print(f"Renamed {len(plan)} image files.")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
