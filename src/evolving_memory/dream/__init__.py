"""Dream engine — 3-phase memory consolidation (SWS -> REM -> Consolidation)."""

from .domain_adapter import DreamDomainAdapter
from .engine import DreamEngine
from .prompt_builder import DreamPromptBuilder

__all__ = ["DreamEngine", "DreamDomainAdapter", "DreamPromptBuilder"]
