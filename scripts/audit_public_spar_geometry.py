"""Audit selected public SPAR questions against generated geometry evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def _available_markers(geometry: Mapping[str, Any]) -> set[str]:
    markers: set[str] = set()
    for view in geometry.get("views", []):
        if isinstance(view, Mapping) and isinstance(view.get("markers"), Mapping):
            markers.update(str(color) for color in view["markers"])
    return markers


def audit_geometry_records(
    questions: list[Mapping[str, Any]],
    geometries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    missing_counts: Counter[str] = Counter()
    unavailable_ids: list[str] = []
    label_conflicts: list[str] = []
    available_count = 0
    rows: list[dict[str, Any]] = []
    for question in questions:
        question_id = str(question.get("question_id"))
        geometry = geometries.get(question_id)
        if geometry is None:
            unavailable_ids.append(question_id)
            rows.append({"question_id": question_id, "status": "missing_geometry"})
            continue
        required = {str(color) for color in geometry.get("required_marker_colors", [])}
        available = _available_markers(geometry)
        missing = sorted(required - available)
        status = str(geometry.get("task_geometry_status", "unknown"))
        label_conflict = False
        if status == "available" and question.get("task") == "distance_infer_center_oo":
            view = next((item for item in geometry.get("views", []) if isinstance(item, Mapping)), {})
            markers = view.get("markers", {}) if isinstance(view, Mapping) else {}
            try:
                distances = {
                    color: math.dist(markers["red"]["camera_xyz"], markers[color]["camera_xyz"])
                    for color in ("green", "blue")
                }
                target = "farther" if "farther" in str(question.get("question", "")).lower() else "closer"
                expected = (max if target == "farther" else min)(distances, key=distances.get)
                answer_text = ""
                for line in str(question.get("question", "")).splitlines():
                    if re.match(r"^[A-D]\.\s*", line) and line[0] == str(question.get("answer", "")).upper():
                        answer_text = line.lower()
                        break
                gold_color = next(
                    (color for color in ("green", "blue") if f"({color} point)" in answer_text),
                    None,
                )
                label_conflict = gold_color != expected
            except (KeyError, TypeError, ValueError):
                label_conflict = True
        if label_conflict:
            label_conflicts.append(question_id)
        if status == "available" and not missing and not label_conflict:
            available_count += 1
        else:
            unavailable_ids.append(question_id)
        missing_counts.update(missing)
        rows.append(
            {
                "question_id": question_id,
                "task": question.get("task"),
                "status": status,
                "unavailable_reason": geometry.get("unavailable_reason"),
                "required_marker_colors": sorted(required),
                "available_marker_colors": sorted(available),
                "missing_marker_colors": missing,
                "label_conflict": label_conflict,
            }
        )
    return {
        "question_count": len(questions),
        "available_count": available_count,
        "unavailable_count": len(questions) - available_count,
        "missing_marker_counts": dict(sorted(missing_counts.items())),
        "unavailable_question_ids": sorted(unavailable_ids),
        "label_conflict_question_ids": sorted(label_conflicts),
        "questions": rows,
    }


def ready_subset(subset: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a subset containing only questions with usable geometry."""
    unavailable = {str(value) for value in report.get("unavailable_question_ids", [])}
    questions = [
        question
        for question in subset.get("questions", [])
        if str(question.get("question_id")) not in unavailable
    ]
    return {
        **dict(subset),
        "questions": questions,
        "question_ids": [str(question.get("question_id")) for question in questions],
        "geometry_audit": {
            "source_question_count": len(subset.get("questions", [])),
            "removed_question_count": len(subset.get("questions", [])) - len(questions),
            "removed_question_ids": sorted(unavailable),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit public SPAR geometry readiness.")
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, default=Path("data/public_spar/geometry_oracle"))
    parser.add_argument("--output", type=Path, default=Path("outputs/public_spar/geometry_audit.json"))
    parser.add_argument("--ready-subset", type=Path, default=None)
    args = parser.parse_args()
    subset = json.loads(args.subset.read_text(encoding="utf-8"))
    questions = subset.get("questions", [])
    geometries = {
        str(question["question_id"]): json.loads(
            (args.geometry_dir / f"{question['question_id']}.json").read_text(encoding="utf-8")
        )
        for question in questions
    }
    report = audit_geometry_records(questions, geometries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.ready_subset is not None:
        filtered = ready_subset(subset, report)
        args.ready_subset.parent.mkdir(parents=True, exist_ok=True)
        args.ready_subset.write_text(
            json.dumps(filtered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.ready_subset}")
    print(f"Audited {report['question_count']} questions")
    print(f"Geometry-ready: {report['available_count']}")
    print(f"Geometry-unavailable: {report['unavailable_count']}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
