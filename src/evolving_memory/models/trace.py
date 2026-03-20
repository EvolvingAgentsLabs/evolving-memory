"""Trace models — raw execution records captured during agent work sessions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ..isa.opcodes import ISA_VERSION
from .hierarchy import HierarchyLevel, TraceOutcome, TraceSource


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class ActionEntry(BaseModel):
    """A single action within a trace."""
    timestamp: datetime = Field(default_factory=_utcnow)
    reasoning: str
    action_payload: str
    result: str = ""


class TraceEntry(BaseModel):
    """One logical unit of work — a trace of actions toward a goal."""
    trace_id: str = Field(default_factory=_new_id)
    hierarchy_level: HierarchyLevel
    parent_trace_id: str | None = None
    goal: str
    outcome: TraceOutcome = TraceOutcome.UNKNOWN
    confidence: float = 0.0
    action_entries: list[ActionEntry] = Field(default_factory=list)
    session_id: str | None = None
    source: TraceSource = TraceSource.UNKNOWN_SOURCE
    tags: list[str] = Field(default_factory=list)
    isa_version: str = ISA_VERSION
    created_at: datetime = Field(default_factory=_utcnow)


class TraceSession(BaseModel):
    """A top-level work session grouping multiple traces."""
    session_id: str = Field(default_factory=_new_id)
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    root_goal: str = ""
    traces: list[TraceEntry] = Field(default_factory=list)
