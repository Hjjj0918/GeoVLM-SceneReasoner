from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_public_spar_prompts import audit_prompt_records
from scripts.build_public_spar_prompts import build_prompt_record


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


def _spatial_question():
    return {
        "question_id": "spar_tiny_000251",
        "dataset": "spar_bench_tiny_rgbd",
        "source_id": "251",
        "images": ["spar_tiny_000251_view_00.jpg"],
        "question": "Where is the red bbox relative to the blue bbox from the observer's viewpoint?",
        "task": "obj_spatial_relation_oo",
        "format_type": "select",
        "img_type": "single_view",
        "answer": "C",
        "evaluation": {"metric": "multiple_choice", "acceptable_answers": "C"},
        "geometry_source": "oracle_rgbd",
    }


def _geometry_000205():
    return json.loads(Path("data/public_spar/geometry_oracle/spar_tiny_000205.json").read_text(encoding="utf-8"))


def _geometry():
    return {
        "question_id": "spar_tiny_000007",
        "geometry_source": "oracle_rgbd",
        "task_geometry_status": "available",
        "views": [
            {
                "view_index": 0,
                "depth": {"valid": True, "valid_pixel_count": 3, "min": 1.0, "max": 3.0, "median": 2.0},
                "intrinsic_color": {"valid": True, "shape": [3, 3]},
                "intrinsic_depth": {"valid": True, "shape": [3, 3]},
                "pose": {"valid": True, "shape": [4, 4]},
                "markers": {
                    "red": {"pixel_xy": [1.5, 2.5], "pixel_bbox": [1, 2, 2, 3], "depth": 2.0, "camera_xyz": [-1.5, -0.5, 2.0]},
                    "green": {"pixel_xy": [2.5, 1.5], "pixel_bbox": [2, 1, 3, 2], "depth": 3.0, "camera_xyz": [0.0, 1.0, 3.0]},
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
            }
        ],
    }


def test_audit_prompt_records_reports_front_loaded_decision_aids():
    records = [
        build_prompt_record(_center_distance_question(), _geometry_000205()),
        build_prompt_record(_spatial_question(), _geometry()),
    ]

    report = audit_prompt_records(records)

    assert report["record_count"] == 2
    assert report["leading_newline_question_ids"] == []
    assert report["missing_front_loaded_decision_aid_question_ids"] == []
    assert report["front_loaded_decision_aid_question_ids"] == ["spar_tiny_000205", "spar_tiny_000251"]


def test_audit_prompt_records_flags_leading_newline():
    record = build_prompt_record(_center_distance_question(), _geometry_000205())
    record["prompts"] = dict(record["prompts"])
    record["prompts"]["geovlm"] = "\n" + record["prompts"]["geovlm"]

    report = audit_prompt_records([record])

    assert report["record_count"] == 1
    assert report["leading_newline_question_ids"] == ["spar_tiny_000205"]
    assert report["missing_front_loaded_decision_aid_question_ids"] == ["spar_tiny_000205"]
