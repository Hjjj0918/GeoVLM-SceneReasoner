from __future__ import annotations

import json
from pathlib import Path

from scripts.build_public_spar_prompts import build_prompt_record
from scripts.check_public_spar_prompt_readiness import check_prompt_readiness


def _center_distance_question(question_id: str = "spar_tiny_000205", answer: str = "A"):
    return {
        "question_id": question_id,
        "dataset": "spar_bench_tiny_rgbd",
        "source_id": question_id.rsplit("_", 1)[-1],
        "images": [f"{question_id}_view_00.jpg"],
        "question": (
            "From the image, decide whether the green point (towel) or blue point (towel) is farther to trash can (red point). "
            "Calculate or judge based on the 3D center points of these objects. Pick the appropriate option from the choices available."
        ),
        "task": "distance_infer_center_oo",
        "format_type": "select",
        "img_type": "single_view",
        "answer": answer,
        "evaluation": {"metric": "multiple_choice", "acceptable_answers": answer},
        "geometry_source": "oracle_rgbd",
    }


def _geometry():
    return {
        "question_id": "spar_tiny_000205",
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
            }
        ],
    }


def test_check_prompt_readiness_reports_ready_for_matching_subset_and_prompts(tmp_path: Path):
    subset_path = tmp_path / "subset.json"
    prompts_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "readiness.json"
    subset_path.write_text(json.dumps({"question_ids": ["spar_tiny_000205"]}), encoding="utf-8")
    prompts_path.write_text(
        json.dumps(build_prompt_record(_center_distance_question(), _geometry()), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = check_prompt_readiness(subset_path, prompts_path, output_path)

    assert report["ready"] is True
    assert report["missing_question_ids"] == []
    assert report["extra_question_ids"] == []
    assert report["leading_newline_question_ids"] == []
    assert report["missing_front_loaded_decision_aid_question_ids"] == []
    assert output_path.exists()


def test_check_prompt_readiness_flags_id_mismatch(tmp_path: Path):
    subset_path = tmp_path / "subset.json"
    prompts_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "readiness.json"
    subset_path.write_text(json.dumps({"question_ids": ["spar_tiny_000205"]}), encoding="utf-8")
    prompts_path.write_text(
        json.dumps(build_prompt_record(_center_distance_question("spar_tiny_000206"), _geometry()), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = check_prompt_readiness(subset_path, prompts_path, output_path)

    assert report["ready"] is False
    assert report["missing_question_ids"] == ["spar_tiny_000205"]
    assert report["extra_question_ids"] == ["spar_tiny_000206"]
