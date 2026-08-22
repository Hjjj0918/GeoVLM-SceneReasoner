"""Shared provider transport helpers for local and public VLM inference."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping


def image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_message_content(payload: Mapping[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Provider response has no choices[0].message.content: {payload}") from error
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    return str(content)


class MockProvider:
    """Return a fixed answer without network access for smoke tests."""

    def __init__(self, model_name: str = "mock-model", response: str = "unknown"):
        self.model_name = model_name
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def answer(self, prompt: str, image_path: Path | None = None) -> str:
        self.calls.append({"prompt": prompt, "image_path": image_path})
        return self.response


class OpenAICompatibleProvider:
    """Call an OpenAI-compatible chat-completions endpoint over HTTP."""

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key_env: str,
        temperature: float = 0.0,
        max_tokens: int = 32,
        timeout_seconds: float = 120.0,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set")
        self.model_name = model_name
        self.endpoint = api_base.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint += "/chat/completions"
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def answer(self, prompt: str, image_path: Path | None = None) -> str:
        content: Any = prompt
        if image_path is not None:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
            ]
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM provider HTTP {error.code}: {body}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"VLM provider connection failed: {error}") from error
        return extract_message_content(response_payload)


__all__ = [
    "MockProvider",
    "OpenAICompatibleProvider",
    "extract_message_content",
    "image_data_url",
]
