"""DreamDomainAdapter protocol — allows different domains to customize LLM prompts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DreamDomainAdapter(Protocol):
    """Protocol for domain-specific dream engine customization.

    Domains (robotics, software agents, etc.) implement this to provide
    custom LLM system prompts for each dream phase.
    """

    @property
    def domain_name(self) -> str:
        """Short identifier for this domain (e.g., 'robotics', 'software')."""
        ...

    def sws_system_prompt(self) -> str:
        """System prompt for Phase 1 (SWS) — failure analysis and critical path."""
        ...

    def rem_system_prompt(self) -> str:
        """System prompt for Phase 2 (REM) — strategy abstraction / node building."""
        ...

    def consolidation_context(self) -> str:
        """Optional extra context for Phase 3 consolidation decisions."""
        ...
