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
                "red": {"pixel_xy": [1.5, 2.5], "pixel_bbox": [1, 2, 2, 3], "depth": 2.0, "camera_xyz": [-1.5, -0.5, 2.0]},
                "green": {"pixel_xy": [2.5, 3.5], "pixel_bbox": [2, 3, 3, 4], "depth": 3.0, "camera_xyz": [0.0, 1.0, 3.0]},
                "blue": {"pixel_xy": [4.5, 4.5], "pixel_bbox": [4, 4, 5, 5], "depth": 5.0, "camera_xyz": [3.75, 3.75, 5.0]},
            },
            "marker_relations": {
                "depth_difference_blue_minus_red": 3.0,
                "euclidean_distance": 7.41,
                "pairwise": {
                    "red_to_green": {"euclidean_distance": 1.414214},
                    "red_to_blue": {"euclidean_distance": 7.41},
                },
            },
        }],
    }


def _spatial_question():
    question = _question()
    question.update(
        {
            "task": "obj_spatial_relation_oo",
            "format_type": "select",
            "answer": "D",
            "evaluation": {"metric": "multiple_choice", "acceptable_answers": "D"},
            "question": (
                "Where is the red bbox relative to the blue bbox from the observer's viewpoint?"
            ),
        }
    )
    return question


def _center_distance_question():
    return {
        "question_id": "spar_tiny_000205",
        "dataset": "spar_bench_tiny_rgbd",
        "source_id": "205",
        "images": ["spar_tiny_000205_view_00.jpg"],
        "question": (
            "From the image, decide whether the green point (towel) or blue point (towel) is farther to trash can (red point). "
            "Calculate or judge based on the 3D center points of these objects. Pick the appropriate option from the choices available."
        ),
        "task": "distance_infer_center_oo",
        "format_type": "select",
        "img_type": "single_view",
        "answer": "A",
        "evaluation": {"metric": "multiple_choice", "acceptable_answers": "A"},
        "geometry_source": "oracle_rgbd",
    }


def _center_distance_geometry():
    with open("data/public_spar/geometry_oracle/spar_tiny_000205.json", "r", encoding="utf-8") as file:
        return json.load(file)


def test_build_prompt_record_separates_evidence_for_all_three_tracks():
    record = build_prompt_record(_question(), _geometry())

    assert set(record["prompts"]) == {"pure_vlm", "geometry_only", "geovlm"}
    assert record["requires_image"] == {"pure_vlm": True, "geometry_only": False, "geovlm": True}
    assert "depth_median" not in record["prompts"]["pure_vlm"].lower()
    assert "depth" in record["prompts"]["geometry_only"].lower()
    assert "red marker" in record["prompts"]["geometry_only"].lower()
    assert "euclidean_distance" in record["prompts"]["geometry_only"].lower()
    assert "green marker" in record["prompts"]["geometry_only"].lower()
    assert "red_to_green" in record["prompts"]["geometry_only"]
    assert "camera_xyz" in record["prompts"]["geometry_only"]
    assert "x increases to the right" in record["prompts"]["geometry_only"].lower()
    assert record["img_type"] == "single_view"
    assert "rgb-d" in record["prompts"]["geovlm"].lower()


def test_numeric_prompt_requires_one_number_and_declares_unit():
    record = build_prompt_record(_question(), _geometry())

    for prompt in record["prompts"].values():
        assert "one number" in prompt.lower() or "numeric" in prompt.lower()
        assert "meter" in prompt.lower()


def test_geovlm_treats_camera_geometry_as_authoritative_for_spatial_tasks():
    record = build_prompt_record(_question(), _geometry())
    prompt = record["prompts"]["geovlm"].lower()

    assert "camera_xyz geometry as authoritative" in prompt
    assert "use the image only to identify" in prompt
    assert "do not override geometric relations" in prompt


def test_geovlm_center_distance_prompt_forbids_z_only_reasoning():
    record = build_prompt_record(_center_distance_question(), _center_distance_geometry())
    prompt = record["prompts"]["geovlm"].lower()

    assert "compare the full 3d euclidean distance in camera_xyz from each candidate marker" in prompt
    assert "do not compare candidate markers to each other" in prompt
    assert "do not use z alone" in prompt
    assert "camera_xyz" in prompt
    assert "red_to_green=447.628657" in prompt
    assert "red_to_blue=680.719033" in prompt
    assert "blue is farther than green" in prompt
    assert "decision aid: red_to_green=447.628657; red_to_blue=680.719033; blue is farther than green." in prompt


def test_geovlm_center_distance_decision_aid_precedes_geometry_block():
    record = build_prompt_record(_center_distance_question(), _center_distance_geometry())
    prompt = record["prompts"]["geovlm"].lower()

    assert prompt.index("decision aid:") < prompt.index("structured oracle rgb-d evidence:")


def test_geovlm_center_distance_prompt_starts_with_decision_aid():
    record = build_prompt_record(_center_distance_question(), _center_distance_geometry())
    prompt = record["prompts"]["geovlm"].lower()

    assert prompt.startswith("decision aid: red_to_green=447.628657; red_to_blue=680.719033; blue is farther than green.")
    assert "final reminder for center-distance questions" not in prompt


def test_geovlm_spatial_relation_prompt_includes_front_loaded_decision_aid():
    record = build_prompt_record(_spatial_question(), _geometry())
    prompt = record["prompts"]["geovlm"].lower()

    assert prompt.startswith("decision aid:")
    assert "decision aid: red is left of blue; red is above blue; red is closer than blue." in prompt
    assert prompt.index("decision aid:") < prompt.index("structured oracle rgb-d evidence:")


def test_spatial_relation_prompt_defines_observer_depth_and_ambiguous_vertical_relation():
    record = build_prompt_record(_spatial_question(), _geometry())

    for track in ("geometry_only", "geovlm"):
        prompt = record["prompts"][track].lower()
        assert "relative to the observer" in prompt
        assert "camera-space z" in prompt
        assert "larger z is farther" in prompt
        assert "do not use euclidean distance between the two markers" in prompt
        assert "use image-space bbox centers for left/right and above/below; larger image y is lower" in prompt
        assert "leave that relation empty" in prompt
        assert "pixel_bbox" in prompt
        assert "euclidean_distance=" not in prompt

    assert "use euclidean distance between camera_xyz points for closer/farther questions" not in record[
        "prompts"
    ]["geometry_only"].lower()


def test_prompt_metadata_keeps_answer_for_offline_evaluation_but_prompt_text_does_not():
    question = _question()
    record = build_prompt_record(question, _geometry())

    assert record["answer"] == "3.5"
    assert all("answer: 3.5" not in prompt.lower() for prompt in record["prompts"].values())
    assert prompt_contains_answer_leak(record["prompts"], question["answer"]) is False


def test_prompt_contains_answer_leak_detects_explicit_gold_key_only():
    assert prompt_contains_answer_leak({"geometry": "gold answer: 3.5"}, "3.5") is True
    assert prompt_contains_answer_leak({"geometry": "median depth: 3.5"}, "3.5") is False


def test_geovlm_non_spatial_prompt_does_not_start_with_newline():
    record = build_prompt_record(_question(), _geometry())
    assert record["prompts"]["geovlm"].startswith("You are answering a spatial reasoning benchmark question using the RGB image and structured RGB-D geometry.")
