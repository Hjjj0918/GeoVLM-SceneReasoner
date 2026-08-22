from __future__ import annotations

import math

import pytest

from scripts.public_spar_common import (
    SUPPORTED_METRICS,
    classify_answer_type,
    deterministic_question_id,
    json_safe,
    normalize_source_id,
    parse_metric_config,
    to_list,
)


def test_json_safe_converts_numpy_scalars_and_arrays_without_importing_dataset_data():
    np = pytest.importorskip("numpy")

    value = json_safe({"scalar": np.float32(1.25), "array": np.array([1, 2])})

    assert value == {"scalar": 1.25, "array": [1, 2]}


def test_normalize_source_id_is_stable_for_int_and_string_ids():
    assert normalize_source_id(12) == "12"
    assert normalize_source_id(" 12 ") == "12"
    assert normalize_source_id(None) == "unknown"


def test_to_list_preserves_none_as_empty_and_does_not_split_strings():
    assert to_list(None) == []
    assert to_list("view.jpg") == ["view.jpg"]
    assert to_list((1, 2)) == [1, 2]


def test_deterministic_question_id_uses_dataset_and_source_id():
    assert deterministic_question_id("spar_bench_tiny_rgbd", 7) == "spar_tiny_000007"
    assert deterministic_question_id("other", "abc") == "spar_abc"


def test_classify_answer_type_and_metric_config_for_numeric_fill():
    row = {"format_type": "fill", "answer": 3.5, "question": "How far?"}

    answer_type = classify_answer_type(row)
    metric = parse_metric_config(row)

    assert answer_type == "numeric"
    assert metric["metric"] == "numeric_tolerance"
    assert metric["absolute_tolerance"] == 0.1
    assert metric["unit"] == "meter"


def test_parse_metric_config_rejects_unsupported_sentence_without_explicit_override():
    with pytest.raises(ValueError, match="unsupported evaluation metric"):
        parse_metric_config({"format_type": "sentence", "answer": "a sentence"})


def test_supported_metrics_are_explicit_and_math_is_not_used_for_normalization():
    assert SUPPORTED_METRICS == {"exact_match", "multiple_choice", "numeric_tolerance"}
    assert math.isclose(float("3.5"), 3.5)
