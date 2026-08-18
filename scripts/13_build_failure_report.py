"""Build a pipeline failure report from baseline results.

Usage: python scripts/13_build_failure_report.py --overwrite
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if isinstance(record, dict):
                records.append(record)
    return records


def sorted_counter(counter: collections.Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def missing_target_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("error_reason") == "missing_target_objects"]


def build_failure_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    missing_records = missing_target_records(records)
    label_counts: collections.Counter[str] = collections.Counter()
    type_counts: collections.Counter[str] = collections.Counter()
    image_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    for record in missing_records:
        image = str(record.get("image", "unknown"))
        image_groups[image].append(record)
        type_counts[str(record.get("type", "unknown"))] += 1
        for label in record.get("missing_target_objects", []):
            label_counts[str(label)] += 1

    image_reports = []
    for image, image_records in sorted(image_groups.items()):
        image_label_counts: collections.Counter[str] = collections.Counter()
        question_ids = []
        for record in image_records:
            question_ids.append(str(record.get("question_id", "")))
            for label in record.get("missing_target_objects", []):
                image_label_counts[str(label)] += 1
        image_reports.append(
            {
                "image": image,
                "missing_record_count": len(image_records),
                "missing_label_counts": sorted_counter(image_label_counts),
                "question_ids": question_ids,
            }
        )

    total = len(records)
    missing_count = len(missing_records)
    return {
        "total_records": total,
        "missing_target_records": missing_count,
        "missing_target_rate": round(missing_count / total, 6) if total else 0.0,
        "missing_label_counts": sorted_counter(label_counts),
        "missing_by_type": sorted_counter(type_counts),
        "images": sorted(image_reports, key=lambda item: (-item["missing_record_count"], item["image"])),
    }


def csv_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for image_report in report.get("images", []):
        label_counts = image_report.get("missing_label_counts", {})
        rows.append(
            {
                "image": str(image_report.get("image", "")),
                "missing_record_count": str(image_report.get("missing_record_count", 0)),
                "missing_labels": ";".join(label_counts.keys()),
                "missing_label_counts": ";".join(f"{label}:{count}" for label, count in label_counts.items()),
                "question_ids": ";".join(image_report.get("question_ids", [])),
            }
        )
    return rows


def write_csv(report: dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = csv_rows(report)
    fieldnames = ["image", "missing_record_count", "missing_labels", "missing_label_counts", "question_ids"]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_failure_report(baseline_path: Path, report_path: Path, csv_path: Path, overwrite: bool) -> int:
    existing = [path for path in (report_path, csv_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Failure report output already exists: {existing[0]}")

    records = load_jsonl(baseline_path)
    report = build_failure_report(records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(report, csv_path)
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GeoVLM pipeline failure report.")
    parser.add_argument("--baseline", type=Path, default=Path("outputs/reasoning/geometry_rule_baseline.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("outputs/evaluations/pipeline_failure_report.json"))
    parser.add_argument("--csv", type=Path, default=Path("outputs/evaluations/pipeline_failure_report.csv"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing failure report outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        count = write_failure_report(
            baseline_path=args.baseline,
            report_path=args.report,
            csv_path=args.csv,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print(f"Wrote {args.report}")
    print(f"Wrote {args.csv}")
    print(f"Read {count} baseline records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
