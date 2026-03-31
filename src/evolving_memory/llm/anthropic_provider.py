"""Anthropic LLM provider adapter."""

from __future__ import annotations

from .base import BaseLLMProvider
from .types import LLMJsonResponse, LLMProgramResponse, extract_json_robust

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

    async def complete_json(self, prompt: str, system: str = "") -> LLMJsonResponse:
        text = await self.complete(prompt, system)
        data = extract_json_robust(text)
        return LLMJsonResponse(raw_text=text, data=data)

    async def emit_program(self, prompt: str, system: str = "") -> LLMProgramResponse:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = await self._client.messages.create(**kwargs)
        return LLMProgramResponse(raw_text=response.content[0].text)
