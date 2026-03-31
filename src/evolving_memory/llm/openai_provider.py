"""OpenAI LLM provider adapter."""

from __future__ import annotations

from .base import BaseLLMProvider
from .types import LLMJsonResponse, LLMProgramResponse, extract_json_robust

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment, misc]


class OpenAIProvider(BaseLLMProvider):
    """Adapter for OpenAI chat completions."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        if AsyncOpenAI is None:
            raise ImportError("openai is required: pip install openai")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    async def complete_json(self, prompt: str, system: str = "") -> LLMJsonResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or "{}"
        data = extract_json_robust(text)
        return LLMJsonResponse(raw_text=text, data=data)

    async def emit_program(self, prompt: str, system: str = "") -> LLMProgramResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0,
        )
        text = response.choices[0].message.content or ""
        return LLMProgramResponse(raw_text=text)
