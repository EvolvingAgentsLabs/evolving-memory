"""Tests for data models."""

import pytest

from evolving_memory.models.hierarchy import (
    HierarchyLevel, TraceOutcome, TraceSource, EdgeType, RouterPath,
)
from evolving_memory.models.trace import ActionEntry, TraceEntry, TraceSession
from evolving_memory.models.graph import ParentNode, ChildNode, ThoughtEdge
from evolving_memory.models.strategy import NegativeConstraint, Strategy, DreamJournalEntry
from evolving_memory.models.query import EntryPoint, RouterDecision, TraversalState


class TestEnums:
    def test_hierarchy_levels(self):
        assert HierarchyLevel.GOAL == 1
        assert HierarchyLevel.REACTIVE == 4
        assert len(HierarchyLevel) == 4

    def test_trace_outcome_values(self):
        assert TraceOutcome.SUCCESS == "success"
        assert TraceOutcome.FAILURE == "failure"

    def test_edge_types(self):
        assert EdgeType.NEXT_STEP == "next_step"
        assert EdgeType.IS_CHILD_OF == "is_child_of"

    def test_router_paths(self):
        assert RouterPath.ZERO_SHOT == "zero_shot"
        assert RouterPath.MEMORY_TRAVERSAL == "memory_traversal"
        assert RouterPath.CONTEXT_JUMP == "context_jump"


class TestTraceModels:
    def test_action_entry_defaults(self):
        a = ActionEntry(reasoning="think", action_payload="do")
        assert a.reasoning == "think"
        assert a.result == ""
        assert a.timestamp is not None

    def test_trace_entry_defaults(self):
        t = TraceEntry(hierarchy_level=HierarchyLevel.TACTICAL, goal="test")
        assert t.trace_id
        assert t.outcome == TraceOutcome.UNKNOWN
        assert t.action_entries == []

    def test_trace_session_defaults(self):
        s = TraceSession()
        assert s.session_id
        assert s.ended_at is None
        assert s.traces == []


class TestGraphModels:
    def test_parent_node_success_rate(self):
        p = ParentNode(
            hierarchy_level=HierarchyLevel.TACTICAL,
            success_count=3,
            failure_count=1,
        )
        assert p.success_rate == 0.75

    def test_parent_node_success_rate_zero(self):
        p = ParentNode(hierarchy_level=HierarchyLevel.TACTICAL)
        assert p.success_rate == 0.0

    def test_child_node_defaults(self):
        c = ChildNode(
            parent_node_id="abc",
            hierarchy_level=HierarchyLevel.TACTICAL,
        )
        assert c.parent_node_id == "abc"
        assert c.is_critical_path is True

    def test_thought_edge(self):
        e = ThoughtEdge(
            source_node_id="a",
            target_node_id="b",
            edge_type=EdgeType.NEXT_STEP,
        )
        assert e.weight == 1.0
        assert e.edge_id


class TestStrategyModels:
    def test_negative_constraint(self):
        nc = NegativeConstraint(
            parent_node_id="p1",
            description="avoid retries without backoff",
        )
        assert nc.constraint_id
        assert nc.parent_node_id == "p1"

    def test_dream_journal_entry(self):
        j = DreamJournalEntry()
        assert j.traces_processed == 0
        assert j.phase_log == []


class TestQueryModels:
    def test_router_decision_zero_shot(self):
        d = RouterDecision(path=RouterPath.ZERO_SHOT)
        assert d.entry_point is None
        assert d.confidence == 0.0

    def test_traversal_state(self):
        s = TraversalState(parent_node_id="p1", total_steps=5)
        assert s.current_child_index == 0
        assert not s.anomaly_detected
