"""Test comparison of Pure VLM, geometry-only, and GeoVLM result files.

Usage: python -m pytest tests/test_compare_track_results.py -q
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_comparison_module():
    module_path = REPO_ROOT / "scripts" / "17_compare_track_results.py"
    spec = importlib.util.spec_from_file_location("compare_track_results", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompareTrackResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.module = load_comparison_module()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_results(self, track: str, rows: list[dict]) -> Path:
        path = self.tmp_path / f"{track}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
        return path

    def row(self, question_id: str, correct: bool, prediction: str = "laptop") -> dict:
        return {
            "question_id": question_id,
            "image": "scene_0001_view_01.jpg",
            "type": "closer_farther",
            "track": "test",
            "model": "test-model",
            "prediction": prediction,
            "raw_response": prediction,
            "correct": correct,
            "error": None,
            "latency_seconds": 1.0,
        }

    def test_build_comparison_calculates_metrics_and_pairwise_counts(self):
        pure_rows = [
            self.row("q001", True),
            self.row("q002", True),
            self.row("q003", False, prediction="mouse"),
        ]
        geovlm_rows = [
            self.row("q001", True),
            self.row("q002", False, prediction="mouse"),
            self.row("q003", True),
        ]
        geometry_rows = [
            self.row("q001", True),
            self.row("q002", False, prediction="mouse"),
            self.row("q003", False, prediction="mouse"),
        ]

        comparison = self.module.build_comparison(
            {
                "pure_vlm": pure_rows,
                "geometry_only": geometry_rows,
                "geovlm": geovlm_rows,
            }
        )

        self.assertEqual(comparison["tracks"]["pure_vlm"]["correct"], 2)
        self.assertEqual(comparison["tracks"]["pure_vlm"]["accuracy"], 0.666667)
        pairwise = comparison["pairwise"]["pure_vlm_vs_geovlm"]
        self.assertEqual(pairwise["both_correct"], 1)
        self.assertEqual(pairwise["left_only_correct"], 1)
        self.assertEqual(pairwise["right_only_correct"], 1)
        self.assertEqual(pairwise["both_wrong"], 0)
        self.assertEqual(len(comparison["disagreements"]), 2)

    def test_build_comparison_rejects_mismatched_question_ids(self):
        records = {
            "pure_vlm": [self.row("q001", True)],
            "geometry_only": [self.row("q001", True)],
            "geovlm": [self.row("q002", True)],
        }

        with self.assertRaises(ValueError):
            self.module.build_comparison(records)


if __name__ == "__main__":
    unittest.main()
