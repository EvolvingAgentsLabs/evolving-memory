"""TraceLogger — context manager + decorator API for capturing execution traces."""

from __future__ import annotations

import functools
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Generator

from ..models.hierarchy import HierarchyLevel, TraceOutcome
from ..models.trace import ActionEntry, TraceEntry


class TraceContext:
    """Mutable context for recording actions within a trace."""

    def __init__(self, trace: TraceEntry) -> None:
        self._trace = trace

    def action(self, reasoning: str, action_payload: str, result: str = "") -> None:
        self._trace.action_entries.append(ActionEntry(
            reasoning=reasoning,
            action_payload=action_payload,
            result=result,
        ))

    def set_outcome(self, outcome: TraceOutcome, confidence: float = 1.0) -> None:
        self._trace.outcome = outcome
        self._trace.confidence = confidence

    def tag(self, *tags: str) -> None:
        self._trace.tags.extend(tags)

    @property
    def trace_id(self) -> str:
        return self._trace.trace_id


class TraceLogger:
    """Captures execution traces via context managers and decorators."""

    def __init__(self, session_id: str | None = None) -> None:
        self._session_id = session_id
        self._traces: list[TraceEntry] = []
        self._trace_stack: list[TraceEntry] = []

    @property
    def traces(self) -> list[TraceEntry]:
        return list(self._traces)

    @contextmanager
    def trace(
        self,
        level: HierarchyLevel,
        goal: str,
        tags: list[str] | None = None,
    ) -> Generator[TraceContext, None, None]:
        parent_id = self._trace_stack[-1].trace_id if self._trace_stack else None
        entry = TraceEntry(
            hierarchy_level=level,
            goal=goal,
            session_id=self._session_id,
            parent_trace_id=parent_id,
            tags=tags or [],
        )
        self._trace_stack.append(entry)
        ctx = TraceContext(entry)
        try:
            yield ctx
            if entry.outcome == TraceOutcome.UNKNOWN:
                entry.outcome = TraceOutcome.SUCCESS
                entry.confidence = 1.0
        except Exception:
            entry.outcome = TraceOutcome.FAILURE
            entry.confidence = 1.0
            raise
        finally:
            self._trace_stack.pop()
            self._traces.append(entry)

    def traced(
        self,
        level: HierarchyLevel,
        goal: str | None = None,
        tags: list[str] | None = None,
    ) -> Callable:
        """Decorator that wraps a function in a trace."""
        def decorator(fn: Callable) -> Callable:
            trace_goal = goal or fn.__name__

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.trace(level, trace_goal, tags) as ctx:
                    result = fn(*args, **kwargs)
                    ctx.action(
                        reasoning=f"Called {fn.__name__}",
                        action_payload=f"args={args}, kwargs={kwargs}",
                        result=str(result) if result is not None else "",
                    )
                    return result
            return wrapper
        return decorator
