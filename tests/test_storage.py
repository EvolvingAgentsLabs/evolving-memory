"""Tests for SQLiteStore and VectorIndex."""

import pytest
import numpy as np

from evolving_memory.models.graph import ParentNode, ChildNode, ThoughtEdge
from evolving_memory.models.hierarchy import HierarchyLevel, TraceOutcome, EdgeType
from evolving_memory.models.strategy import NegativeConstraint
from evolving_memory.storage.sqlite_store import SQLiteStore
from evolving_memory.storage.vector_index import VectorIndex

from conftest import make_parent_node, make_child_node, make_session


class TestSQLiteStore:
    def test_save_and_get_parent_node(self, store):
        node = make_parent_node(goal="implement JWT", summary="JWT strategy")
        store.save_parent_node(node)
        retrieved = store.get_parent_node(node.node_id)
        assert retrieved is not None
        assert retrieved.goal == "implement JWT"
        assert retrieved.summary == "JWT strategy"

    def test_get_all_parent_nodes(self, store):
        store.save_parent_node(make_parent_node(goal="goal A"))
        store.save_parent_node(make_parent_node(goal="goal B"))
        nodes = store.get_all_parent_nodes()
        assert len(nodes) == 2

    def test_increment_access(self, store):
        node = make_parent_node()
        store.save_parent_node(node)
        assert store.get_parent_node(node.node_id).access_count == 0
        store.increment_access(node.node_id)
        assert store.get_parent_node(node.node_id).access_count == 1

    def test_save_and_get_child_nodes(self, store):
        parent = make_parent_node()
        store.save_parent_node(parent)
        c0 = make_child_node(parent.node_id, step_index=0)
        c1 = make_child_node(parent.node_id, step_index=1)
        store.save_child_node(c0)
        store.save_child_node(c1)
        children = store.get_child_nodes_for_parent(parent.node_id)
        assert len(children) == 2
        assert children[0].step_index == 0
        assert children[1].step_index == 1

    def test_save_and_get_edges(self, store):
        edge = ThoughtEdge(
            source_node_id="a", target_node_id="b", edge_type=EdgeType.NEXT_STEP
        )
        store.save_edge(edge)
        from_edges = store.get_edges_from("a")
        to_edges = store.get_edges_to("b")
        assert len(from_edges) == 1
        assert len(to_edges) == 1
        assert from_edges[0].edge_type == EdgeType.NEXT_STEP

    def test_save_session_and_retrieve_unprocessed(self, store):
        session = make_session(root_goal="build auth", n_traces=2, n_actions=3)
        store.save_session(session)
        unprocessed = store.get_unprocessed_sessions()
        assert len(unprocessed) == 1
        assert unprocessed[0].root_goal == "build auth"
        assert len(unprocessed[0].traces) == 2
        assert len(unprocessed[0].traces[0].action_entries) == 3

    def test_mark_session_processed(self, store):
        session = make_session()
        store.save_session(session)
        store.mark_session_processed(session.session_id)
        assert store.get_unprocessed_sessions() == []

    def test_save_negative_constraint(self, store):
        parent = make_parent_node()
        store.save_parent_node(parent)
        nc = NegativeConstraint(
            parent_node_id=parent.node_id,
            description="avoid retries",
        )
        store.save_negative_constraint(nc)
        constraints = store.get_constraints_for_parent(parent.node_id)
        assert len(constraints) == 1
        assert constraints[0].description == "avoid retries"


class TestVectorIndex:
    def test_add_and_search(self, vector_index):
        vec = np.random.randn(768).astype(np.float32)
        vector_index.add("node_1", vec)
        results = vector_index.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "node_1"
        assert results[0][1] > 0.99  # Same vector, high similarity

    def test_search_empty_index(self, vector_index):
        vec = np.random.randn(768).astype(np.float32)
        results = vector_index.search(vec, top_k=5)
        assert results == []

    def test_multiple_nodes(self, vector_index):
        for i in range(10):
            vec = np.random.randn(768).astype(np.float32)
            vector_index.add(f"node_{i}", vec)
        assert vector_index.size == 10
        query = np.random.randn(768).astype(np.float32)
        results = vector_index.search(query, top_k=3)
        assert len(results) == 3

    def test_remove_node(self, vector_index):
        vec = np.random.randn(768).astype(np.float32)
        vector_index.add("to_remove", vec)
        assert vector_index.size == 1
        vector_index.remove("to_remove")
        assert vector_index.size == 0
