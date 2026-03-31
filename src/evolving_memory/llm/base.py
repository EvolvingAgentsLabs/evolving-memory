"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import LLMJsonResponse, LLMProgramResponse


class BaseLLMProvider(ABC):
    """Interface for LLM providers used by the dream engine."""

    @abstractmethod
    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the LLM and return the completion text."""
        ...

    @abstractmethod
    async def complete_json(self, prompt: str, system: str = "") -> LLMJsonResponse:
        """Send a prompt expecting a JSON response, parse and return as LLMJsonResponse."""
        ...

    @abstractmethod
    async def emit_program(self, prompt: str, system: str = "") -> LLMProgramResponse:
        """Send a prompt expecting ISA opcode output, return LLMProgramResponse.

        Unlike complete_json(), the raw_text is passed to the ISA parser.
        Providers should use temperature=0.0 for deterministic emission.
        """
        ...
