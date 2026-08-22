from __future__ import annotations

import json

from scripts.build_public_spar_prompts import build_prompt_record, prompt_contains_answer_leak


def _question():
    return {
        "question_id": "spar_tiny_000007",
        "dataset": "spar_bench_tiny_rgbd",
        "source_id": "7",
        "images": ["spar_tiny_000007_view_00.jpg"],
        "question": "What is the distance in meters?",
        "task": "distance_prediction_oc",
        "format_type": "fill",
        "img_type": "single_view",
        "answer": "3.5",
        "answer_type": "numeric",
        "evaluation": {"metric": "numeric_tolerance", "unit": "meter", "absolute_tolerance": 0.1},
        "geometry_source": "oracle_rgbd",
    }


def _geometry():
    return {
        "question_id": "spar_tiny_000007",
        "geometry_source": "oracle_rgbd",
        "task_geometry_status": "available",
        "views": [{
            "view_index": 0,
            "depth": {"valid": True, "valid_pixel_count": 3, "min": 1.0, "max": 3.0, "median": 2.0},
            "intrinsic_color": {"valid": True, "shape": [3, 3]},
            "intrinsic_depth": {"valid": True, "shape": [3, 3]},
            "pose": {"valid": True, "shape": [4, 4]},
            "markers": {
                "red": {"pixel_xy": [1.5, 2.5], "depth": 2.0, "camera_xyz": [-1.5, -0.5, 2.0]},
                "blue": {"pixel_xy": [4.5, 4.5], "depth": 5.0, "camera_xyz": [3.75, 3.75, 5.0]},
            },
            "marker_relations": {"depth_difference_blue_minus_red": 3.0, "euclidean_distance": 7.41},
        }],
    }


def test_build_prompt_record_separates_evidence_for_all_three_tracks():
    record = build_prompt_record(_question(), _geometry())

    assert set(record["prompts"]) == {"pure_vlm", "geometry_only", "geovlm"}
    assert record["requires_image"] == {"pure_vlm": True, "geometry_only": False, "geovlm": True}
    assert "depth_median" not in record["prompts"]["pure_vlm"].lower()
    assert "depth" in record["prompts"]["geometry_only"].lower()
    assert "red marker" in record["prompts"]["geometry_only"].lower()
    assert "euclidean_distance" in record["prompts"]["geometry_only"].lower()
    assert "rgb-d" in record["prompts"]["geovlm"].lower()


def test_numeric_prompt_requires_one_number_and_declares_unit():
    record = build_prompt_record(_question(), _geometry())

    for prompt in record["prompts"].values():
        assert "one number" in prompt.lower() or "numeric" in prompt.lower()
        assert "meter" in prompt.lower()


def test_prompt_metadata_keeps_answer_for_offline_evaluation_but_prompt_text_does_not():
    question = _question()
    record = build_prompt_record(question, _geometry())

    assert record["answer"] == "3.5"
    assert all("answer: 3.5" not in prompt.lower() for prompt in record["prompts"].values())
    assert prompt_contains_answer_leak(record["prompts"], question["answer"]) is False


def test_prompt_contains_answer_leak_detects_explicit_gold_key_only():
    assert prompt_contains_answer_leak({"geometry": "gold answer: 3.5"}, "3.5") is True
    assert prompt_contains_answer_leak({"geometry": "median depth: 3.5"}, "3.5") is False
