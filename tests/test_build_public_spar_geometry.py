from __future__ import annotations

import json

import numpy as np

from scripts.build_public_spar_geometry import (
    build_geometry_record,
    build_geometry_files_streaming,
    repair_geometry_file,
    contains_answer_leak,
    extract_marker_geometry,
    _sample_depth,
    validate_intrinsics,
    validate_pose,
    required_marker_colors,
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
    assert validate_intrinsics(np.eye(4).reshape(-1))["valid"] is True
    assert validate_pose(np.eye(4).reshape(-1))["valid"] is True


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
    record = build_geometry_record(
        row,
        {
            "question_id": "spar_tiny_000007",
            "question": row["question"],
        },
    )

    assert record["task_geometry_status"] == "geometry_unavailable"
    assert record["unavailable_reason"] == "no_point_or_object_reference"
    assert "9.0" not in json.dumps(record)


def test_required_marker_colors_reads_colors_from_question_text():
    assert required_marker_colors("Which is closer: chair (green point) or box (blue point) to bag (red point)?") == {
        "red",
        "green",
        "blue",
    }
    assert required_marker_colors("Compare the laptop (red bbox) with monitor (blue bbox).") == {
        "red",
        "blue",
    }


def test_marker_task_is_unavailable_when_a_required_marker_has_no_depth():
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[1:3, 1:3] = [255, 0, 0]
    image[4:6, 4:6] = [0, 0, 255]
    row = {
        **_row(),
        "task": "distance_infer_center_oo",
        "question": "Which is closer: chair (green point) or box (blue point) to bag (red point)?",
        "image": [image],
        "depth": [np.zeros((7, 7), dtype=np.float32)],
    }

    record = build_geometry_record(
        row,
        {"question_id": "spar_tiny_000007", "question": row["question"]},
    )

    assert record["task_geometry_status"] == "geometry_unavailable"
    assert record["unavailable_reason"] == "required_marker_missing"
    assert record["required_marker_colors"] == ["blue", "green", "red"]


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
    assert markers["red"]["pixel_bbox"] == [1, 2, 2, 3]


def test_extract_marker_geometry_uses_bbox_center_when_overlapping_markers_occlude_pixels():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[2, 2:18] = [255, 0, 0]
    image[12, 2:18] = [255, 0, 0]
    image[2:13, 2] = [255, 0, 0]
    image[2:13, 17] = [255, 0, 0]
    image[1, 10:28] = [0, 0, 255]
    image[11, 10:28] = [0, 0, 255]
    image[1:12, 10] = [0, 0, 255]
    image[1:12, 27] = [0, 0, 255]
    depth = np.full((20, 30), 5.0, dtype=np.float32)
    intrinsics = np.array([[2.0, 0.0, 15.0], [0.0, 2.0, 10.0], [0.0, 0.0, 1.0]])

    markers = extract_marker_geometry(image, depth, intrinsics)

    assert markers["red"]["pixel_bbox"] == [2, 2, 17, 12]
    assert markers["blue"]["pixel_bbox"] == [10, 1, 27, 11]
    assert markers["red"]["pixel_xy"] == [9.5, 7.0]
    assert markers["blue"]["pixel_xy"] == [18.5, 6.0]
    assert markers["red"]["visible_pixel_median_xy"] != markers["red"]["pixel_xy"]


def test_extract_marker_geometry_reshapes_flattened_four_by_four_intrinsics():
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[2:4, 1:3] = [255, 0, 0]
    depth = np.full((7, 7), 4.0, dtype=np.float32)
    intrinsics = np.array(
        [2.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )

    markers = extract_marker_geometry(image, depth, intrinsics)

    assert markers["red"]["camera_xyz"] is not None


def test_extract_marker_geometry_ignores_saturated_wood_background_when_finding_red_marker():
    image = np.full((20, 20, 3), [190, 105, 70], dtype=np.uint8)
    image[8:12, 14:18] = [255, 0, 0]
    image[2:6, 2:6] = [0, 0, 255]
    depth = np.full((20, 20), 3.0, dtype=np.float32)

    markers = extract_marker_geometry(image, depth, np.eye(3, dtype=np.float32))

    assert markers["red"]["pixel_xy"] == [15.5, 9.5]
    assert markers["red"]["marker_pixel_count"] == 16


def test_extract_marker_geometry_keeps_low_brightness_green_annotation():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[4:8, 4:8] = [255, 0, 0]
    image[9:13, 9:13] = [0, 128, 1]
    image[14:18, 14:18] = [0, 0, 255]
    depth = np.full((20, 20), 2.0, dtype=np.float32)

    markers = extract_marker_geometry(image, depth, np.eye(3, dtype=np.float32))

    assert set(markers) == {"red", "green", "blue"}
    assert markers["green"]["pixel_xy"] == [10.5, 10.5]


def test_sample_depth_falls_back_to_valid_pixels_near_marker_center():
    depth = np.zeros((9, 9), dtype=np.float32)
    depth[4, 6] = 7.0

    value, count = _sample_depth(depth, 4, 4)

    assert value == 7.0
    assert count == 1


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


def test_marker_relations_include_all_available_marker_pairs():
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[1:3, 1:3] = [255, 0, 0]
    image[3:5, 3:5] = [0, 255, 0]
    image[5:7, 5:7] = [0, 0, 255]
    row = {
        **_row(),
        "image": [image],
        "depth": [np.full((7, 7), 2.0, dtype=np.float32)],
        "intrinsic_depth": [np.eye(3, dtype=np.float32)],
    }

    record = build_geometry_record(row, {"question_id": "spar_tiny_000007"})
    pairwise = record["views"][0]["marker_relations"]["pairwise"]

    assert "red_to_green" in pairwise
    assert "red_to_blue" in pairwise
    assert "green_to_blue" in pairwise


def test_build_geometry_files_streaming_writes_only_selected_source_rows(tmp_path, monkeypatch):
    rows = []
    for source_id in range(5):
        row = _row()
        row["id"] = source_id
        rows.append(row)
    question_records = [
        {"question_id": "spar_tiny_000001", "source_id": "1", "images": ["one.jpg"]},
        {"question_id": "spar_tiny_000003", "source_id": "3", "images": ["three.jpg"]},
    ]
    calls = []

    def fake_iter_dataset_rows(*, columns=None, **kwargs):
        calls.append(columns)
        yield from rows

    monkeypatch.setattr("scripts.build_public_spar_geometry.iter_dataset_rows", fake_iter_dataset_rows)

    paths = build_geometry_files_streaming(
        dataset_name="fixture",
        split="test",
        question_records=question_records,
        output_dir=tmp_path,
        streaming=True,
        overwrite=False,
    )

    assert {path.name for path in paths} == {"spar_tiny_000001.json", "spar_tiny_000003.json"}
    assert "image" in calls[0]
    assert "depth" in calls[0]
    assert json.loads((tmp_path / "spar_tiny_000001.json").read_text())["source_id"] == "1"


def test_repair_geometry_file_uses_flattened_or_4x4_intrinsics_for_marker_xyz(tmp_path):
    path = tmp_path / "sample.json"
    payload = {
        "views": [{
            "intrinsic_depth": {"valid": True, "shape": [4, 4], "values": [
                [2.0, 0.0, 1.0, 0.0], [0.0, 2.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]
            ]},
            "markers": {
                "red": {"depth_pixel_xy": [3.0, 5.0], "depth": 4.0, "camera_xyz": None},
                "blue": {"depth_pixel_xy": [5.0, 5.0], "depth": 6.0, "camera_xyz": None},
            },
            "marker_relations": {},
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    repair_geometry_file(path)

    repaired = json.loads(path.read_text(encoding="utf-8"))
    assert repaired["views"][0]["markers"]["red"]["camera_xyz"] == [4.0, 8.0, 4.0]
    assert repaired["views"][0]["marker_relations"]["euclidean_distance"] > 0
