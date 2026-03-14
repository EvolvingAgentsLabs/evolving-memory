"""Dream engine — 3-phase memory consolidation (SWS → REM → Consolidation)."""

from .domain_adapter import DreamDomainAdapter
from .engine import DreamEngine

__all__ = ["DreamEngine", "DreamDomainAdapter"]
