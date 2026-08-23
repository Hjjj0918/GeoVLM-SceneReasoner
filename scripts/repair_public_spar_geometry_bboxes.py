"""Add marker pixel bounding boxes to existing local SPAR geometry files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_public_spar_geometry import _camera_xyz, _marker_mask, _marker_relations


def repair_geometry_bboxes(geometry_dir: Path, images_dir: Path) -> int:
    updated = 0
    for path in sorted(geometry_dir.glob("spar_tiny_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for view in payload.get("views", []):
            image_name = view.get("image_name")
            image = cv2.imread(str(images_dir / str(image_name))) if image_name else None
            if image is None or not isinstance(view.get("markers"), dict):
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for color, marker in view["markers"].items():
                if not isinstance(marker, dict):
                    continue
                ys, xs = np.where(_marker_mask(image_rgb, str(color)))
                if xs.size == 0:
                    continue
                marker["visible_pixel_median_xy"] = [
                    round(float(np.median(xs)), 3),
                    round(float(np.median(ys)), 3),
                ]
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                if marker.get("pixel_bbox") != bbox:
                    marker["pixel_bbox"] = bbox
                    changed = True
                min_x, min_y, max_x, max_y = bbox
                image_x = (min_x + max_x) / 2.0
                image_y = (min_y + max_y) / 2.0
                marker["pixel_xy"] = [round(image_x, 3), round(image_y, 3)]
                depth_shape = view.get("depth", {}).get("shape", [])
                intrinsic_values = view.get("intrinsic_depth", {}).get("values")
                if isinstance(depth_shape, list) and len(depth_shape) >= 2:
                    depth_y = image_y * float(depth_shape[0]) / max(image.shape[0], 1)
                    depth_x = image_x * float(depth_shape[1]) / max(image.shape[1], 1)
                    marker["depth_pixel_xy"] = [round(depth_x, 3), round(depth_y, 3)]
                    if isinstance(intrinsic_values, list) and isinstance(marker.get("depth"), (int, float)):
                        intrinsic = np.asarray(intrinsic_values, dtype=float)
                        marker["camera_xyz"] = _camera_xyz(
                            depth_x, depth_y, float(marker["depth"]), intrinsic
                        )
                changed = True
            view["marker_relations"] = _marker_relations(view["markers"])
        if changed:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            updated += 1
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair local SPAR geometry marker bboxes.")
    parser.add_argument("--geometry-dir", type=Path, default=Path("data/public_spar/geometry_oracle"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/public_spar/images"))
    args = parser.parse_args()
    updated = repair_geometry_bboxes(args.geometry_dir, args.images_dir)
    print(f"Updated {updated} geometry files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
