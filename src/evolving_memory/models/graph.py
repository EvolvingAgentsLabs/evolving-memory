"""Thought graph models — the consolidated memory structure."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ..isa.opcodes import ISA_VERSION
from .hierarchy import HierarchyLevel, TraceOutcome, EdgeType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class ThoughtNode(BaseModel):
    """Base node in the thought graph."""
    node_id: str = Field(default_factory=_new_id)
    hierarchy_level: HierarchyLevel
    content: str = ""
    summary: str = ""
    confidence: float = 0.0
    access_count: int = 0
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ParentNode(ThoughtNode):
    """High-level node representing a consolidated goal/strategy. ONLY parent nodes get embeddings."""
    child_node_ids: list[str] = Field(default_factory=list)
    embedding_vector: list[float] | None = None
    goal: str = ""
    outcome: TraceOutcome = TraceOutcome.UNKNOWN
    trigger_goals: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    version: int = 1
    isa_version: str = ISA_VERSION

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total


class ChildNode(ThoughtNode):
    """A single step within a parent's strategy — reasoning + action + result."""
    parent_node_id: str = ""
    step_index: int = 0
    reasoning: str = ""
    action: str = ""
    result: str = ""
    is_critical_path: bool = True


class ThoughtEdge(BaseModel):
    """Directed edge in the thought graph."""
    edge_id: str = Field(default_factory=_new_id)
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    weight: float = 1.0
    created_at: datetime = Field(default_factory=_utcnow)
