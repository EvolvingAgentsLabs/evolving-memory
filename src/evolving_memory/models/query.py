"""Query models — used by the cognitive router for memory retrieval."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .hierarchy import RouterPath
from .graph import ParentNode


class EntryPoint(BaseModel):
    """A candidate entry point found via semantic search."""
    parent_node: ParentNode
    similarity_score: float = 0.0
    composite_score: float = 0.0


class RouterDecision(BaseModel):
    """The router's decision on how to handle a query."""
    path: RouterPath
    reasoning: str = ""
    entry_point: EntryPoint | None = None
    confidence: float = 0.0


class TraversalState(BaseModel):
    """Mutable state for step-by-step traversal through a parent's children."""
    parent_node_id: str
    current_child_index: int = 0
    total_steps: int = 0
    anomaly_detected: bool = False
