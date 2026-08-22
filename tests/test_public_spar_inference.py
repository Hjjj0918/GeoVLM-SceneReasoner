from __future__ import annotations

import json
from pathlib import Path

from scripts.public_spar_inference import (
    evaluate_prediction,
    normalize_public_prediction,
    summarize_public_results,
)


def test_normalize_multiple_choice_accepts_canonical_label_only():
    assert normalize_public_prediction("The answer is option B.", {"metric": "multiple_choice"}) == "B"
    assert normalize_public_prediction("not sure", {"metric": "multiple_choice"}) == "unknown"


def test_normalize_multiple_choice_keeps_a_wrong_but_parseable_label():
    assert normalize_public_prediction(
        "B", {"metric": "multiple_choice", "acceptable_answers": "A"}
    ) == "B"


def test_normalize_multiple_choice_rejects_non_benchmark_labels_and_unknown_text():
    config = {"metric": "multiple_choice"}
    assert normalize_public_prediction("I don't know", config) == "unknown"
    assert normalize_public_prediction("option E", config) == "unknown"


def test_normalize_numeric_strips_units_and_rejects_empty_response():
    config = {"metric": "numeric_tolerance", "unit": "meter", "absolute_tolerance": 0.1}

    assert normalize_public_prediction("3.50 meters", config) == "3.5"
    assert normalize_public_prediction("", config) == "unknown"


def test_evaluate_numeric_uses_record_tolerance_boundary():
    record = {"evaluation": {"metric": "numeric_tolerance", "absolute_tolerance": 0.1}, "answer": "3.5"}

    assert evaluate_prediction("3.6", record)["correct"] is True
    assert evaluate_prediction("3.61", record)["correct"] is False


def test_summarize_public_results_reports_parse_failures_and_by_task():
    records = [
        {"task": "distance", "format_type": "fill", "img_type": "single_view", "prediction": "A", "correct": True, "parse_status": "parsed"},
        {"task": "distance", "format_type": "fill", "img_type": "single_view", "prediction": "unknown", "correct": False, "parse_status": "parse_failure"},
    ]

    summary = summarize_public_results(records, track="pure_vlm", model="mock")

    assert summary["total"] == 2
    assert summary["parse_failures"] == 1
    assert summary["coverage"] == 0.5
    assert summary["by_task"]["distance"]["total"] == 2
    json.dumps(summary)


def test_run_public_inference_attaches_images_only_to_visual_tracks(tmp_path: Path):
    from scripts.public_spar_inference import MockProvider, run_public_inference_file

    prompts_path = tmp_path / "prompts.jsonl"
    subset_path = tmp_path / "subset.json"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "q1.jpg"
    image_path.write_bytes(b"image")
    record = {
        "question_id": "q1",
        "images": ["q1.jpg"],
        "task": "distance",
        "format_type": "select",
        "img_type": "single_view",
        "evaluation": {"metric": "multiple_choice", "acceptable_answers": "A"},
        "answer": "A",
        "prompts": {"pure_vlm": "image", "geometry_only": "geometry", "geovlm": "both"},
    }
    prompts_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    subset_path.write_text(json.dumps({"question_ids": ["q1"]}), encoding="utf-8")

    geometry_provider = MockProvider(response="A")
    run_public_inference_file(
        prompt_path=prompts_path,
        subset_path=subset_path,
        images_dir=images_dir,
        output_path=tmp_path / "geometry.jsonl",
        summary_path=tmp_path / "geometry.json",
        track="geometry_only",
        provider=geometry_provider,
        overwrite=False,
    )
    assert geometry_provider.calls[0]["image_path"] is None

    visual_provider = MockProvider(response="A")
    run_public_inference_file(
        prompt_path=prompts_path,
        subset_path=subset_path,
        images_dir=images_dir,
        output_path=tmp_path / "pure.jsonl",
        summary_path=tmp_path / "pure.json",
        track="pure_vlm",
        provider=visual_provider,
        overwrite=False,
    )
    assert visual_provider.calls[0]["image_path"] == image_path
