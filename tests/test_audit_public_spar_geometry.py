from __future__ import annotations

from scripts.audit_public_spar_geometry import audit_geometry_records, ready_subset


def test_audit_geometry_records_reports_required_marker_gaps():
    questions = [
        {
            "question_id": "q1",
            "task": "distance_infer_center_oo",
            "question": "Compare red point, green point, and blue point.",
        },
        {
            "question_id": "q2",
            "task": "obj_spatial_relation_oo",
            "question": "Compare red bbox and blue bbox.",
        },
    ]
    geometries = {
        "q1": {
            "task_geometry_status": "geometry_unavailable",
            "unavailable_reason": "required_marker_missing",
            "required_marker_colors": ["blue", "green", "red"],
            "views": [{"markers": {"red": {}, "blue": {}}}],
        },
        "q2": {
            "task_geometry_status": "available",
            "required_marker_colors": ["blue", "red"],
            "views": [{"markers": {"red": {}, "blue": {}}}],
        },
    }

    report = audit_geometry_records(questions, geometries)

    assert report["question_count"] == 2
    assert report["available_count"] == 1
    assert report["unavailable_count"] == 1
    assert report["missing_marker_counts"] == {"green": 1}
    assert report["unavailable_question_ids"] == ["q1"]


def test_ready_subset_removes_only_geometry_unavailable_questions():
    subset = {"phase": "a", "questions": [{"question_id": "q1"}, {"question_id": "q2"}]}
    report = {"unavailable_question_ids": ["q1"]}

    filtered = ready_subset(subset, report)

    assert [row["question_id"] for row in filtered["questions"]] == ["q2"]
    assert filtered["question_ids"] == ["q2"]


def test_audit_marks_distance_gold_conflict_from_camera_xyz():
    questions = [{
        "question_id": "q1",
        "task": "distance_infer_center_oo",
        "question": "Which is closer: chair (green point) or box (blue point) to bag (red point)?\nA. chair (green point)\nB. box (blue point)",
        "answer": "A",
    }]
    geometries = {"q1": {
        "task_geometry_status": "available",
        "required_marker_colors": ["blue", "green", "red"],
        "views": [{"markers": {
            "red": {"camera_xyz": [0, 0, 0]},
            "green": {"camera_xyz": [0, 0, 10]},
            "blue": {"camera_xyz": [0, 0, 2]},
        }}],
    }}

    report = audit_geometry_records(questions, geometries)

    assert report["label_conflict_question_ids"] == ["q1"]
    assert report["unavailable_question_ids"] == ["q1"]
