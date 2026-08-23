from __future__ import annotations

import json

import cv2
import numpy as np

from scripts.repair_public_spar_geometry_bboxes import repair_geometry_bboxes


def test_repair_geometry_bboxes_adds_local_marker_bbox(tmp_path):
    geometry_dir = tmp_path / "geometry"
    images_dir = tmp_path / "images"
    geometry_dir.mkdir()
    images_dir.mkdir()

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2:4, 1:3] = [0, 0, 255]
    cv2.imwrite(str(images_dir / "view.png"), image)
    geometry_path = geometry_dir / "spar_tiny_000001.json"
    geometry_path.write_text(
        json.dumps(
            {
                "views": [
                    {
                        "image_name": "view.png",
                        "depth": {"shape": [4, 4]},
                        "intrinsic_depth": {
                            "values": [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]
                        },
                        "markers": {"red": {"pixel_xy": [1.5, 2.5], "depth": 4.0}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert repair_geometry_bboxes(geometry_dir, images_dir) == 1
    repaired = json.loads(geometry_path.read_text(encoding="utf-8"))
    assert repaired["views"][0]["markers"]["red"]["pixel_bbox"] == [1, 2, 2, 3]
    assert repaired["views"][0]["markers"]["red"]["pixel_xy"] == [1.5, 2.5]
    assert repaired["views"][0]["markers"]["red"]["visible_pixel_median_xy"] == [1.5, 2.5]
    assert repaired["views"][0]["markers"]["red"]["depth_pixel_xy"] == [0.75, 1.25]
    assert repaired["views"][0]["markers"]["red"]["camera_xyz"] == [-0.5, 0.5, 4.0]
