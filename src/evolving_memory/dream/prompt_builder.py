"""Structured prompt assembly for dream phases — ported from claw-code SystemPromptBuilder.

Replaces ad-hoc string concatenation with a composable builder that
supports domain adapters, fidelity context, and negative constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .domain_adapter import DreamDomainAdapter


class DreamPromptBuilder:
    """Composable prompt builder for dream engine LLM calls.

    Usage:
        prompt = (
            DreamPromptBuilder()
            .with_phase("SWS")
            .with_domain_context(adapter, "sws")
            .append_section("Trace", trace_text)
            .with_negative_constraints(constraints)
            .build()
        )
    """

    def __init__(self) -> None:
        self._sections: list[str] = []

    def with_phase(self, phase_name: str) -> "DreamPromptBuilder":
        """Insert phase header at the start."""
        self._sections.insert(0, f"# Dream Phase: {phase_name}")
        return self

    def with_isa_version(self, version: str) -> "DreamPromptBuilder":
        """Declare the ISA version for this prompt."""
        self._sections.append(f"ISA Version: {version}")
        return self

    def with_domain_context(self, adapter: "DreamDomainAdapter", phase: str) -> "DreamPromptBuilder":
        """Inject domain-specific context from a DreamDomainAdapter.

        Args:
            adapter: The domain adapter providing context.
            phase: One of "sws", "rem", "consolidation".
        """
        dispatch = {
            "sws": adapter.sws_system_prompt,
            "rem": adapter.rem_system_prompt,
            "consolidation": adapter.consolidation_context,
        }
        method = dispatch.get(phase)
        if method:
            text = method()
            if text:
                self._sections.append(f"# Domain Context ({adapter.domain_name})\n{text}")
        return self

    def with_negative_constraints(self, constraints: list[str]) -> "DreamPromptBuilder":
        """Add negative constraints section if any exist."""
        if constraints:
            items = "\n".join(f"- {c}" for c in constraints)
            self._sections.append(f"# Negative Constraints\n{items}")
        return self

    def with_fidelity_context(self, source: str, weight: float) -> "DreamPromptBuilder":
        """Add source fidelity metadata."""
        self._sections.append(f"# Source Fidelity\n- Source: {source} (weight: {weight})")
        return self

    def append_section(self, title: str, content: str) -> "DreamPromptBuilder":
        """Append a named section."""
        self._sections.append(f"# {title}\n{content}")
        return self

    def append_raw(self, text: str) -> "DreamPromptBuilder":
        """Append raw text without a section header."""
        self._sections.append(text)
        return self

    def build(self) -> str:
        """Assemble all sections into the final prompt string."""
        return "\n\n".join(self._sections)
