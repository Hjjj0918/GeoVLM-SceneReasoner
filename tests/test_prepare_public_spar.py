from __future__ import annotations

import json

from PIL import Image

from scripts.prepare_public_spar import (
    DEFAULT_SELECTION,
    convert_row,
    export_row_images,
    select_rows,
    selection_audit,
    write_phase_outputs,
)
from scripts.inspect_public_spar_test_support import make_fixture_rows


def test_select_rows_is_deterministic_and_applies_single_view_supported_task_policy():
    rows = list(reversed(make_fixture_rows()))
    selected, excluded = select_rows(rows, phase="a", config=DEFAULT_SELECTION)

    assert [row["id"] for row in selected] == ["2", 1]
    assert len(excluded) == 1
    assert excluded[0]["source_id"] == "3"
    assert "img_type_not_allowed" in excluded[0]["reasons"]


def test_phase_limits_are_explicit():
    rows = [
        {
            **make_fixture_rows()[0],
            "id": index,
            "task": "distance_prediction_oc" if index < 25 else "depth_prediction_oc",
        }
        for index in range(50)
    ]

    selected, _ = select_rows(rows, phase="a", config=DEFAULT_SELECTION)

    assert len(selected) == 30


def test_convert_row_builds_public_schema_and_no_geometry_answer_field():
    record = convert_row(make_fixture_rows()[0])

    assert record["question_id"] == "spar_tiny_000001"
    assert record["images"] == ["spar_tiny_000001_view_00.jpg"]
    assert record["answer_type"] == "numeric"
    assert record["evaluation"] == {
        "metric": "numeric_tolerance",
        "absolute_tolerance": 0.1,
        "unit": "meter",
    }
    assert record["geometry_source"] == "oracle_rgbd"
    assert "geometry_summary" not in record


def test_export_row_images_writes_deterministic_rgb_files(tmp_path):
    row = {**make_fixture_rows()[0], "image": [Image.new("RGB", (3, 2), color="red")]}

    names = export_row_images(row, tmp_path)

    assert names == ["spar_tiny_000001_view_00.jpg"]
    with Image.open(tmp_path / names[0]) as image:
        assert image.size == (3, 2)
        assert image.mode == "RGB"


def test_selection_audit_contains_counts_and_reasons():
    selected, excluded = select_rows(make_fixture_rows(), phase="a", config=DEFAULT_SELECTION)

    audit = selection_audit(selected, excluded, phase="a", config=DEFAULT_SELECTION)

    assert audit["selected_count"] == 2
    assert audit["excluded_count"] == 1
    assert audit["excluded_by_reason"]["img_type_not_allowed"] == 1
    assert json.loads(json.dumps(audit))["phase"] == "a"


def test_write_phase_outputs_writes_unified_questions_file_and_source_fingerprint(tmp_path):
    from PIL import Image

    rows = [{**make_fixture_rows()[0], "image": [Image.new("RGB", (2, 2), "blue")]}]
    subset_path, audit_path = write_phase_outputs(
        rows,
        phase="a",
        output_root=tmp_path / "subsets",
        images_dir=tmp_path / "images",
        questions_path=tmp_path / "questions.spar.json",
        config={**DEFAULT_SELECTION, "per_task_limit": 20},
        overwrite=False,
    )

    assert subset_path.exists()
    assert audit_path.exists()
    unified = json.loads((tmp_path / "questions.spar.json").read_text())
    audit = json.loads(audit_path.read_text())
    assert unified["questions"][0]["question_id"] == "spar_tiny_000001"
    assert audit["selected_source_ids"] == ["1"]
    assert len(audit["source_fingerprint_sha256"]) == 64
