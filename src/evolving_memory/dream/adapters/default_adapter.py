"""Default domain adapter — generic software agent behavior (current prompts)."""

from __future__ import annotations


class DefaultAdapter:
    """Generic software agent adapter. Uses the existing ISA-based prompts."""

    @property
    def domain_name(self) -> str:
        return "software"

    def sws_system_prompt(self) -> str:
        return (
            "You are a cognitive instruction emitter for a memory consolidation system.\n"
            "Output ONLY instructions, one per line. No prose, no explanation.\n"
            "Lines starting with # are comments (optional)."
        )

    def rem_system_prompt(self) -> str:
        return (
            "You are a cognitive instruction emitter for a memory consolidation system.\n"
            "Output ONLY instructions, one per line. No prose, no explanation.\n"
            "Lines starting with # are comments (optional)."
        )

    def consolidation_context(self) -> str:
        return ""
