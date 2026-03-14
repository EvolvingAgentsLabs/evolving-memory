"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Interface for LLM providers used by the dream engine."""

    @abstractmethod
    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the LLM and return the completion text."""
        ...

    @abstractmethod
    async def complete_json(self, prompt: str, system: str = "") -> dict:
        """Send a prompt expecting a JSON response, parse and return as dict."""
        ...

    @abstractmethod
    async def emit_program(self, prompt: str, system: str = "") -> str:
        """Send a prompt expecting ISA opcode output, return raw text for parsing.

        Unlike complete_json(), this returns the raw text so the ISA parser
        can process it.  Providers should use temperature=0.0 for deterministic
        emission.
        """
        ...
