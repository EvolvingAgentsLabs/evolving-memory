"""Dream engine — 4-phase memory consolidation (SWS -> REM -> Consolidation -> Compaction)."""

from .compactor import MemoryCompactor
from .domain_adapter import DreamDomainAdapter
from .engine import DreamEngine
from .prompt_builder import DreamPromptBuilder

__all__ = ["DreamEngine", "DreamDomainAdapter", "DreamPromptBuilder", "MemoryCompactor"]
