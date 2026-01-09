"""Adapter for OpenAI Responses API."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class OpenAIGeneration:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int


class OpenAIModelAdapter:
    def __init__(self, model: str = "gpt-4o-mini", timeout_s: int = 30) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self._client = OpenAI()

    def generate(self, prompt: str) -> OpenAIGeneration:
        start = time.monotonic()
        response = self._client.responses.create(
            model=self.model,
            input=prompt,
            timeout=self.timeout_s,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        text = response.output_text
        tokens_in = response.usage.input_tokens if response.usage else 0
        tokens_out = response.usage.output_tokens if response.usage else 0
        return OpenAIGeneration(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
