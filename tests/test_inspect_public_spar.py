from __future__ import annotations

import json

from scripts.inspect_public_spar_test_support import make_fixture_rows
from scripts import inspect_public_spar


def test_summarize_rows_counts_tasks_formats_views_and_missing_rgbd_fields():
    summary = inspect_public_spar.summarize_rows(make_fixture_rows())

    assert summary["total_rows"] == 3
    assert summary["task_counts"] == {"distance_prediction_oc": 2, "spatial_relation_oo": 1}
    assert summary["format_type_counts"] == {"fill": 2, "select": 1}
    assert summary["img_type_counts"] == {"single_view": 2, "multi_view": 1}
    assert summary["view_count"] == {"1": 2, "2": 1}
    assert summary["missing_field_counts"]["pose"] == 1


def test_build_preview_is_json_safe_and_does_not_include_full_arrays():
    preview = inspect_public_spar.build_preview(make_fixture_rows(), limit=2)

    assert len(preview) == 2
    assert preview[0]["source_id"] == "1"
    assert preview[0]["image"]["type"] == "list"
    assert preview[0]["image"]["length"] == 1
    assert "pixels" not in json.dumps(preview)
    assert "answer" in preview[0]


def test_write_inspection_outputs_creates_json_csv_and_preview(tmp_path):
    paths = inspect_public_spar.write_inspection_outputs(
        make_fixture_rows(), output_dir=tmp_path, preview_limit=2, overwrite=False
    )

    assert {path.name for path in paths} == {
        "inspection_summary.json",
        "task_distribution.csv",
        "sample_preview.json",
    }
    assert json.loads((tmp_path / "inspection_summary.json").read_text())[
        "total_rows"
    ] == 3
    assert (tmp_path / "task_distribution.csv").read_text().count("distance_prediction_oc") == 1
