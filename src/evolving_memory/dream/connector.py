"""Phase 3 — Consolidation: edges, embeddings, merge detection."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..config import DreamConfig
from ..embeddings.encoder import EmbeddingEncoder
from ..isa.parser import InstructionParser
from ..llm.base import BaseLLMProvider
from ..llm.prompts import CONSOLIDATION_LINK_DISCOVERY, CONSOLIDATION_LINK_SYSTEM
from ..models.fidelity import get_fidelity_weight
from ..models.graph import ThoughtEdge, ParentNode
from ..models.hierarchy import EdgeType, TraceOutcome, TraceSource
from ..storage.sqlite_store import SQLiteStore
from ..storage.vector_index import VectorIndex
from ..vm.machine import CognitiveVM
from .chunker import ChunkedResult

logger = logging.getLogger(__name__)


class TopologicalConnector:
    """Phase 3 Consolidation — creates edges, embeddings, and detects merges."""

    def __init__(
        self,
        store: SQLiteStore,
        index: VectorIndex,
        encoder: EmbeddingEncoder,
        config: DreamConfig,
        llm: BaseLLMProvider | None = None,
    ) -> None:
        self._store = store
        self._index = index
        self._encoder = encoder
        self._config = config
        self._llm = llm
        self._parser = InstructionParser()

    async def consolidate(self, chunks: list[ChunkedResult]) -> dict:
        """Consolidate chunked results into the thought graph. Returns stats."""
        stats = {
            "nodes_created": 0,
            "nodes_merged": 0,
            "edges_created": 0,
            "cross_edges_created": 0,
        }
        new_node_ids: list[str] = []

        for chunk in chunks:
            merged = self._try_merge(chunk.parent, chunk.source)
            if merged:
                # Merged into existing node
                stats["nodes_merged"] += 1
                parent = merged
            else:
                # New node — generate embedding and add to index
                embedding = self._encoder.encode(chunk.parent.summary)
                chunk.parent.embedding_vector = embedding.tolist()
                self._store.save_parent_node(chunk.parent)
                self._index.add(chunk.parent.node_id, embedding)
                stats["nodes_created"] += 1
                parent = chunk.parent
                new_node_ids.append(parent.node_id)

            # Save child nodes
            for child in chunk.children:
                child.parent_node_id = parent.node_id
                self._store.save_child_node(child)

            # Update parent's child list
            existing_children = self._store.get_child_nodes_for_parent(parent.node_id)
            parent.child_node_ids = [c.node_id for c in existing_children]
            self._store.save_parent_node(parent)

            # Create edges
            edges_created = self._create_edges(parent, chunk.children)
            stats["edges_created"] += edges_created

        # Cross-trace link discovery (requires LLM)
        if new_node_ids and self._llm is not None:
            cross_edges = await self._discover_cross_links(new_node_ids)
            stats["cross_edges_created"] = cross_edges
            stats["edges_created"] += cross_edges

        return stats

    def _try_merge(
        self,
        candidate: ParentNode,
        source: TraceSource = TraceSource.UNKNOWN_SOURCE,
    ) -> ParentNode | None:
        """Check if a similar parent node exists and merge if so."""
        embedding = self._encoder.encode(candidate.summary)
        results = self._index.search(embedding, top_k=1)
        if not results:
            return None

        best_id, best_score = results[0]
        if best_score < self._config.merge_similarity_threshold:
            return None

        existing = self._store.get_parent_node(best_id)
        if existing is None:
            return None

        # Merge: update existing node with fidelity-weighted confidence boost
        weight = get_fidelity_weight(source)
        existing.version += 1
        existing.confidence = min(1.0, existing.confidence + 0.1 * weight)
        existing.access_count += 1
        if candidate.outcome == TraceOutcome.SUCCESS:
            existing.success_count += candidate.success_count
        if candidate.outcome == TraceOutcome.FAILURE:
            existing.failure_count += candidate.failure_count
        # Merge trigger goals
        for tg in candidate.trigger_goals:
            if tg not in existing.trigger_goals:
                existing.trigger_goals.append(tg)
        # Merge negative constraints
        for nc in candidate.negative_constraints:
            if nc not in existing.negative_constraints:
                existing.negative_constraints.append(nc)
        existing.updated_at = datetime.now(timezone.utc)
        self._store.save_parent_node(existing)
        return existing

    def _create_edges(self, parent: ParentNode, children: list) -> int:
        count = 0
        child_ids = [c.node_id for c in children]

        for i, child in enumerate(children):
            # child → is_child_of → parent
            self._store.save_edge(ThoughtEdge(
                source_node_id=child.node_id,
                target_node_id=parent.node_id,
                edge_type=EdgeType.IS_CHILD_OF,
            ))
            count += 1

            # Temporal edges between sequential children
            if i > 0:
                # previous child → next_step → this child
                self._store.save_edge(ThoughtEdge(
                    source_node_id=child_ids[i - 1],
                    target_node_id=child.node_id,
                    edge_type=EdgeType.NEXT_STEP,
                ))
                count += 1
                # this child → previous_step → previous child
                self._store.save_edge(ThoughtEdge(
                    source_node_id=child.node_id,
                    target_node_id=child_ids[i - 1],
                    edge_type=EdgeType.PREVIOUS_STEP,
                ))
                count += 1

        return count

    async def _discover_cross_links(self, new_node_ids: list[str]) -> int:
        """Find cross-trace connections between newly created and existing nodes."""
        edges_created = 0

        for node_id in new_node_ids:
            node = self._store.get_parent_node(node_id)
            if node is None:
                continue
            embedding = self._encoder.encode(node.summary)

            # Semantic similarity as candidate filter only
            candidates = self._index.search(
                embedding, top_k=self._config.cross_link_top_k
            )

            for candidate_id, sim_score in candidates:
                if candidate_id == node_id:
                    continue
                if sim_score < self._config.cross_link_similarity_floor:
                    continue

                candidate = self._store.get_parent_node(candidate_id)
                if candidate is None:
                    continue

                children_a = self._store.get_child_nodes_for_parent(node_id)
                children_b = self._store.get_child_nodes_for_parent(candidate_id)

                steps_a = "; ".join(
                    f"[{c.step_index}] {c.action}" for c in children_a
                )
                steps_b = "; ".join(
                    f"[{c.step_index}] {c.action}" for c in children_b
                )

                # LLM decides the actual relationship
                raw = await self._llm.emit_program(
                    CONSOLIDATION_LINK_DISCOVERY.format(
                        id_a=node_id,
                        goal_a=node.goal,
                        summary_a=node.summary,
                        steps_a=steps_a or "(none)",
                        id_b=candidate_id,
                        goal_b=candidate.goal,
                        summary_b=candidate.summary,
                        steps_b=steps_b or "(none)",
                    ),
                    system=CONSOLIDATION_LINK_SYSTEM,
                )

                program = self._parser.parse(raw)
                vm = CognitiveVM()
                result = vm.execute(program)

                for edge_data in result.built_edges:
                    edge_type = EdgeType(edge_data["edge_type"])
                    self._store.save_edge(ThoughtEdge(
                        source_node_id=edge_data["source_id"],
                        target_node_id=edge_data["target_id"],
                        edge_type=edge_type,
                        weight=sim_score,
                    ))
                    edges_created += 1
                    logger.debug(
                        "Cross-link: %s → %s (%s, sim=%.3f)",
                        edge_data["source_id"][:8],
                        edge_data["target_id"][:8],
                        edge_type.value,
                        sim_score,
                    )

        return edges_created
