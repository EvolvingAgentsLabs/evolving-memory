"""Anthropic LLM provider adapter."""

from __future__ import annotations

import json
import re

from .base import BaseLLMProvider

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None  # type: ignore[assignment, misc]


class AnthropicProvider(BaseLLMProvider):
    """Adapter for Anthropic messages API."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None) -> None:
        if AsyncAnthropic is None:
            raise ImportError("anthropic is required: pip install anthropic")
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def complete(self, prompt: str, system: str = "") -> str:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        text = await self.complete(prompt, system)
        # Extract JSON from markdown code blocks if present
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
        return json.loads(text)

    async def emit_program(self, prompt: str, system: str = "") -> str:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = await self._client.messages.create(**kwargs)
        return response.content[0].text
