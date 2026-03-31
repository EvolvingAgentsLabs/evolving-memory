"""Gemini LLM provider via OpenAI-compatible API, with vision support."""

from __future__ import annotations

import os

from .base import BaseLLMProvider
from .types import LLMJsonResponse, LLMProgramResponse, extract_json_robust

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment, misc]

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiProvider(BaseLLMProvider):
    """Adapter for Gemini via its OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise ImportError("openai is required: pip install openai")
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("GEMINI_API_KEY", ""),
            base_url=_GEMINI_BASE_URL,
        )
        self._model = model

    async def complete(self, prompt: str, system: str = "") -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    async def complete_json(self, prompt: str, system: str = "") -> LLMJsonResponse:
        messages: list[dict] = []
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
        messages: list[dict] = []
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

    async def complete_vision(
        self,
        prompt: str,
        image_b64: str,
        system: str = "",
        media_type: str = "image/jpeg",
    ) -> str:
        """Send a prompt with a base64-encoded image and return the completion."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_b64}",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        })
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content or ""
