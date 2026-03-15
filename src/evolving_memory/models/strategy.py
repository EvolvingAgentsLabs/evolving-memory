"""Strategy and dream journal models — metadata produced by the dream engine."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class NegativeConstraint(BaseModel):
    """A learned 'do NOT do this' rule extracted from failure traces."""
    constraint_id: str = Field(default_factory=_new_id)
    parent_node_id: str
    description: str
    source_trace_id: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class Strategy(BaseModel):
    """A reusable strategy extracted from successful traces."""
    strategy_id: str = Field(default_factory=_new_id)
    parent_node_id: str
    goal: str
    steps: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)


class DreamJournalEntry(BaseModel):
    """A record of a single dream cycle execution."""
    journal_id: str = Field(default_factory=_new_id)
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    traces_processed: int = 0
    nodes_created: int = 0
    nodes_merged: int = 0
    edges_created: int = 0
    cross_edges_created: int = 0
    constraints_extracted: int = 0
    phase_log: list[str] = Field(default_factory=list)
