from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from scripts.vlm_inference_common import (
    MockProvider,
    extract_message_content,
    image_data_url,
)


def test_image_data_url_contains_detected_mime_and_base64_payload(tmp_path: Path):
    path = tmp_path / "sample.jpg"
    path.write_bytes(b"image-bytes")

    value = image_data_url(path)

    assert value.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(value.split(",", 1)[1]) == b"image-bytes"


def test_extract_message_content_handles_text_parts_and_rejects_missing_choices():
    payload = {"choices": [{"message": {"content": [{"text": "A"}, {"text": "B"}]}}]}
    assert extract_message_content(payload) == "A\nB"
    with pytest.raises(ValueError, match="no choices"):
        extract_message_content({})


def test_mock_provider_records_prompt_and_optional_image(tmp_path: Path):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"x")
    provider = MockProvider(model_name="test", response="A")

    assert provider.answer("question", image) == "A"
    assert provider.calls == [{"prompt": "question", "image_path": image}]
    json.dumps(provider.calls, default=str)
