from __future__ import annotations

import json

import numpy as np

from scripts.build_public_spar_geometry import (
    build_geometry_record,
    contains_answer_leak,
    extract_marker_geometry,
    validate_intrinsics,
    validate_pose,
)


def _row():
    return {
        "id": 7,
        "image": ["rgb.jpg"],
        "depth": [np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)],
        "pose": [np.eye(4, dtype=np.float32)],
        "intrinsic_color": [np.eye(3, dtype=np.float32)],
        "intrinsic_depth": [np.eye(3, dtype=np.float32)],
        "question": "What is the distance?",
        "answer": "3.5",
        "task": "distance_prediction_oc",
    }


def test_validate_camera_matrices_accepts_expected_shapes_and_rejects_invalid_values():
    assert validate_intrinsics(np.eye(3))["valid"] is True
    assert validate_intrinsics(np.eye(3))["shape"] == [3, 3]
    assert validate_pose(np.eye(4))["valid"] is True
    assert validate_pose(np.eye(4))["shape"] == [4, 4]
    assert validate_intrinsics([[1, 2], [3, 4]])["valid"] is False
    assert validate_pose([[1, 2, 3]])["valid"] is False


def test_build_geometry_record_contains_depth_stats_camera_evidence_and_no_answer():
    record = build_geometry_record(
        _row(),
        {"question_id": "spar_tiny_000007", "images": ["spar_tiny_000007_view_00.jpg"]},
    )

    assert record["question_id"] == "spar_tiny_000007"
    assert record["geometry_source"] == "oracle_rgbd"
    assert record["views"][0]["image_name"] == "spar_tiny_000007_view_00.jpg"
    assert record["views"][0]["depth"]["valid_pixel_count"] == 3
    assert record["views"][0]["depth"]["min"] == 1.0
    assert record["views"][0]["depth"]["max"] == 3.0
    assert record["views"][0]["intrinsic_color"]["valid"] is True
    assert record["views"][0]["pose"]["valid"] is True
    assert "\"answer\"" not in json.dumps(record).lower()


def test_contains_answer_leak_detects_gold_answer_in_nested_geometry():
    assert contains_answer_leak({"nested": {"gold_answer": "3.5"}}) is True
    assert contains_answer_leak({"depth": {"max": 3.5}}) is False


def test_unlocalizable_point_task_is_marked_geometry_unavailable_without_inventing_point():
    row = {**_row(), "task": "depth_prediction_oc", "answer": "9.0"}
    record = build_geometry_record(row, {"question_id": "spar_tiny_000007"})

    assert record["task_geometry_status"] == "geometry_unavailable"
    assert record["unavailable_reason"] == "no_point_or_object_reference"
    assert "9.0" not in json.dumps(record)


def test_extract_marker_geometry_reads_red_blue_depth_and_camera_coordinates():
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[2:4, 1:3] = [255, 0, 0]
    image[4:6, 4:6] = [0, 0, 255]
    depth = np.full((7, 7), 5.0, dtype=np.float32)
    depth[1:5, 0:4] = 2.0
    intrinsics = np.array([[2.0, 0.0, 3.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]])

    markers = extract_marker_geometry(image, depth, intrinsics)

    assert set(markers) == {"red", "blue"}
    assert markers["red"]["depth"] == 2.0
    assert markers["blue"]["depth"] == 5.0
    assert len(markers["red"]["camera_xyz"]) == 3


def test_build_geometry_record_makes_marker_task_available_and_adds_pairwise_distance():
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[2:4, 1:3] = [255, 0, 0]
    image[4:6, 4:6] = [0, 0, 255]
    row = {
        **_row(),
        "image": [image],
        "depth": [np.arange(49, dtype=np.float32).reshape(7, 7) + 1],
        "intrinsic_depth": [np.eye(3, dtype=np.float32)],
    }

    record = build_geometry_record(row, {"question_id": "spar_tiny_000007"})

    assert record["task_geometry_status"] == "available"
    assert set(record["views"][0]["markers"]) == {"red", "blue"}
    assert record["views"][0]["marker_relations"]["euclidean_distance"] > 0
