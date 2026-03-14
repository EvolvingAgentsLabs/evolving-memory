"""CognitiveRouter — tripartite decision maker (Zero-Shot / Memory Traversal / Context Jump)."""

from __future__ import annotations

from ..config import RouterConfig
from ..embeddings.encoder import EmbeddingEncoder
from ..models.graph import ChildNode, ParentNode
from ..models.hierarchy import RouterPath
from ..models.query import EntryPoint, RouterDecision, TraversalState
from ..storage.sqlite_store import SQLiteStore
from ..storage.vector_index import VectorIndex
from .anomaly import AnomalyDetector


class CognitiveRouter:
    """Routes queries through the tripartite decision path and manages traversal."""

    def __init__(
        self,
        store: SQLiteStore,
        index: VectorIndex,
        encoder: EmbeddingEncoder,
        config: RouterConfig,
    ) -> None:
        self._store = store
        self._index = index
        self._encoder = encoder
        self._config = config
        self._anomaly = AnomalyDetector(encoder, config.anomaly_threshold)

    def query(self, query_text: str) -> RouterDecision:
        """Route a query: returns a RouterDecision with the chosen path."""
        query_vec = self._encoder.encode(query_text)

        # FAISS search → top-k parent nodes
        results = self._index.search(query_vec, top_k=self._config.top_k)
        if not results:
            return RouterDecision(
                path=RouterPath.ZERO_SHOT,
                reasoning="No memory nodes found — using zero-shot",
                confidence=0.0,
            )

        # Score each candidate
        candidates: list[EntryPoint] = []
        for node_id, sim_score in results:
            parent = self._store.get_parent_node(node_id)
            if parent is None:
                continue
            composite = (
                self._config.similarity_weight * sim_score
                + self._config.confidence_weight * parent.confidence
                + self._config.success_rate_weight * parent.success_rate
            )
            candidates.append(EntryPoint(
                parent_node=parent,
                similarity_score=sim_score,
                composite_score=composite,
            ))

        if not candidates:
            return RouterDecision(
                path=RouterPath.ZERO_SHOT,
                reasoning="No valid parent nodes found — using zero-shot",
                confidence=0.0,
            )

        # Sort by composite score
        candidates.sort(key=lambda c: c.composite_score, reverse=True)
        best = candidates[0]

        if best.composite_score < self._config.composite_threshold:
            return RouterDecision(
                path=RouterPath.ZERO_SHOT,
                reasoning=f"Best composite score {best.composite_score:.3f} below threshold "
                          f"{self._config.composite_threshold} — using zero-shot",
                confidence=best.composite_score,
            )

        # Memory traversal
        self._store.increment_access(best.parent_node.node_id)
        return RouterDecision(
            path=RouterPath.MEMORY_TRAVERSAL,
            reasoning=f"Found relevant memory (composite={best.composite_score:.3f})",
            entry_point=best,
            confidence=best.composite_score,
        )

    def begin_traversal(self, entry_point: EntryPoint) -> TraversalState:
        """Start step-by-step traversal of a parent's children."""
        children = self._store.get_child_nodes_for_parent(entry_point.parent_node.node_id)
        return TraversalState(
            parent_node_id=entry_point.parent_node.node_id,
            current_child_index=0,
            total_steps=len(children),
        )

    def next_step(self, state: TraversalState) -> tuple[ChildNode | None, TraversalState]:
        """Load the next child node in the traversal sequence."""
        if state.current_child_index >= state.total_steps:
            return None, state

        children = self._store.get_child_nodes_for_parent(state.parent_node_id)
        if state.current_child_index >= len(children):
            return None, state

        child = children[state.current_child_index]
        state.current_child_index += 1
        return child, state

    def check_anomaly(self, state: TraversalState, current_context: str) -> TraversalState:
        """Check for semantic drift during traversal."""
        parent = self._store.get_parent_node(state.parent_node_id)
        if parent is None:
            return state
        anomaly, _score = self._anomaly.check(parent.goal, current_context)
        state.anomaly_detected = anomaly
        return state
