from __future__ import annotations

import pytest

from scripts.public_spar_comparison import build_public_comparison


def _record(question_id: str, track: str, correct: bool, prediction: str) -> dict:
    return {
        "question_id": question_id,
        "track": track,
        "model": "mock",
        "task": "distance_infer_center_oo",
        "format_type": "select",
        "img_type": "single_view",
        "prediction": prediction,
        "correct": correct,
        "parse_status": "parsed",
        "raw_response": prediction,
    }


def test_build_public_comparison_reports_tracks_and_pairwise_cases():
    records = {
        track: [_record("q1", track, track != "pure_vlm", "B" if track != "pure_vlm" else "A")]
        for track in ("pure_vlm", "geometry_only", "geovlm")
    }

    comparison = build_public_comparison(records)

    assert comparison["question_count"] == 1
    assert comparison["tracks"]["geovlm"]["accuracy"] == 1.0
    pair = comparison["pairwise"]["pure_vlm_vs_geovlm"]
    assert pair["right_only_correct"] == 1
    assert comparison["disagreement_count"] == 1


def test_build_public_comparison_rejects_mismatched_question_ids():
    records = {
        "pure_vlm": [_record("q1", "pure_vlm", True, "B")],
        "geometry_only": [_record("q2", "geometry_only", True, "B")],
        "geovlm": [_record("q1", "geovlm", True, "B")],
    }

    with pytest.raises(ValueError, match="question IDs differ"):
        build_public_comparison(records)


def test_build_public_comparison_rejects_mismatched_evaluation_definitions():
    records = {
        track: [_record("q1", track, True, "B")]
        for track in ("pure_vlm", "geometry_only", "geovlm")
    }
    records["geovlm"][0]["metric"] = "numeric_tolerance"

    with pytest.raises(ValueError, match="evaluation definitions differ"):
        build_public_comparison(records)
