"""Tests for trace capture — TraceLogger and SessionManager."""

import pytest

from evolving_memory.capture.trace_logger import TraceLogger
from evolving_memory.capture.session import SessionManager
from evolving_memory.models.hierarchy import HierarchyLevel, TraceOutcome
from evolving_memory.storage.sqlite_store import SQLiteStore


class TestTraceLogger:
    def test_basic_trace(self):
        logger = TraceLogger(session_id="s1")
        with logger.trace(HierarchyLevel.TACTICAL, "implement auth") as ctx:
            ctx.action("analyze", "read requirements", result="understood")
            ctx.action("code", "write auth.py", result="200 lines")

        assert len(logger.traces) == 1
        trace = logger.traces[0]
        assert trace.goal == "implement auth"
        assert trace.outcome == TraceOutcome.SUCCESS
        assert len(trace.action_entries) == 2

    def test_failed_trace(self):
        logger = TraceLogger()
        with pytest.raises(ValueError):
            with logger.trace(HierarchyLevel.TACTICAL, "bad op") as ctx:
                ctx.action("try", "something", result="error")
                raise ValueError("boom")

        trace = logger.traces[0]
        assert trace.outcome == TraceOutcome.FAILURE

    def test_explicit_outcome(self):
        logger = TraceLogger()
        with logger.trace(HierarchyLevel.TACTICAL, "partial work") as ctx:
            ctx.action("start", "begin work", result="half done")
            ctx.set_outcome(TraceOutcome.PARTIAL, confidence=0.5)

        assert logger.traces[0].outcome == TraceOutcome.PARTIAL
        assert logger.traces[0].confidence == 0.5

    def test_nested_traces(self):
        logger = TraceLogger()
        with logger.trace(HierarchyLevel.GOAL, "build app") as outer:
            outer.action("plan", "design architecture", result="planned")
            with logger.trace(HierarchyLevel.TACTICAL, "impl module") as inner:
                inner.action("code", "write module.py", result="done")

        assert len(logger.traces) == 2
        inner_trace = logger.traces[0]  # inner finishes first
        outer_trace = logger.traces[1]
        assert inner_trace.parent_trace_id == outer_trace.trace_id

    def test_tags(self):
        logger = TraceLogger()
        with logger.trace(HierarchyLevel.TACTICAL, "tagged op") as ctx:
            ctx.tag("auth", "jwt")

        assert logger.traces[0].tags == ["auth", "jwt"]

    def test_decorator(self):
        logger = TraceLogger()

        @logger.traced(HierarchyLevel.REACTIVE, goal="compute sum")
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5
        assert len(logger.traces) == 1
        assert logger.traces[0].goal == "compute sum"
        assert logger.traces[0].outcome == TraceOutcome.SUCCESS


class TestSessionManager:
    def test_session_persists_traces(self, store):
        mgr = SessionManager(store)
        with mgr.session("build auth") as logger:
            with logger.trace(HierarchyLevel.TACTICAL, "impl JWT") as ctx:
                ctx.action("code", "write jwt.py", result="done")
            with logger.trace(HierarchyLevel.TACTICAL, "test JWT") as ctx:
                ctx.action("test", "run tests", result="pass")

        sessions = store.get_unprocessed_sessions()
        assert len(sessions) == 1
        assert sessions[0].root_goal == "build auth"
        assert len(sessions[0].traces) == 2
