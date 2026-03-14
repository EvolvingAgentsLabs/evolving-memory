"""Tests for the cognitive router and anomaly detector."""

import pytest
import numpy as np

from evolving_memory.config import RouterConfig
from evolving_memory.models.hierarchy import RouterPath, HierarchyLevel, TraceOutcome
from evolving_memory.models.query import EntryPoint
from evolving_memory.models.graph import ParentNode
from evolving_memory.router.cognitive_router import CognitiveRouter
from evolving_memory.router.anomaly import AnomalyDetector

from conftest import MockEmbeddingEncoder, make_parent_node, make_child_node


class TestCognitiveRouter:
    def _setup_router(self, store, vector_index):
        encoder = MockEmbeddingEncoder()
        config = RouterConfig()
        return CognitiveRouter(store, vector_index, encoder, config), encoder

    def test_query_empty_index_returns_zero_shot(self, store, vector_index):
        router, _ = self._setup_router(store, vector_index)
        decision = router.query("how to implement JWT?")
        assert decision.path == RouterPath.ZERO_SHOT

    def test_query_with_memory_returns_traversal(self, store, vector_index):
        router, encoder = self._setup_router(store, vector_index)

        # Add a parent node to store + index
        parent = make_parent_node(goal="implement JWT", summary="JWT auth strategy", confidence=0.9)
        store.save_parent_node(parent)
        vec = encoder.encode(parent.summary)
        vector_index.add(parent.node_id, vec)

        # Query with the same text — should find it
        decision = router.query("JWT auth strategy")
        assert decision.path == RouterPath.MEMORY_TRAVERSAL
        assert decision.entry_point is not None
        assert decision.entry_point.parent_node.goal == "implement JWT"

    def test_traversal(self, store, vector_index):
        router, encoder = self._setup_router(store, vector_index)

        parent = make_parent_node(goal="build feature")
        store.save_parent_node(parent)
        c0 = make_child_node(parent.node_id, step_index=0)
        c1 = make_child_node(parent.node_id, step_index=1)
        c2 = make_child_node(parent.node_id, step_index=2)
        store.save_child_node(c0)
        store.save_child_node(c1)
        store.save_child_node(c2)

        entry = EntryPoint(
            parent_node=parent,
            similarity_score=0.9,
            composite_score=0.8,
        )
        state = router.begin_traversal(entry)
        assert state.total_steps == 3

        children = []
        while True:
            child, state = router.next_step(state)
            if child is None:
                break
            children.append(child)

        assert len(children) == 3
        assert children[0].step_index == 0
        assert children[2].step_index == 2


class TestAnomalyDetector:
    def test_no_anomaly_similar_texts(self):
        encoder = MockEmbeddingEncoder()
        detector = AnomalyDetector(encoder, threshold=0.3)
        # Same text should have high similarity
        anomaly, score = detector.check("implement JWT auth", "implement JWT auth")
        assert not anomaly
        assert score > 0.9

    def test_anomaly_divergent_texts(self):
        encoder = MockEmbeddingEncoder()
        detector = AnomalyDetector(encoder, threshold=0.99)
        # Very high threshold should trigger anomaly for different texts
        anomaly, score = detector.check(
            "implement JWT authentication",
            "cook a delicious pasta carbonara",
        )
        assert anomaly
