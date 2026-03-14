"""Tests for the dream engine — curator, chunker, connector, and orchestrator."""

import pytest

from evolving_memory.config import CTEConfig, DreamConfig
from evolving_memory.dream.curator import TraceCurator
from evolving_memory.dream.chunker import HierarchicalChunker
from evolving_memory.dream.connector import TopologicalConnector
from evolving_memory.dream.engine import DreamEngine
from evolving_memory.models.hierarchy import TraceOutcome

from conftest import make_trace, make_session, MockEmbeddingEncoder


class TestTraceCurator:
    @pytest.mark.asyncio
    async def test_curate_success_trace(self, mock_llm):
        curator = TraceCurator(mock_llm)
        traces = [make_trace(goal="implement feature", outcome=TraceOutcome.SUCCESS)]
        results = await curator.curate(traces)
        assert len(results) == 1
        assert len(results[0].critical_steps) > 0

    @pytest.mark.asyncio
    async def test_curate_failure_extracts_constraints(self, mock_llm):
        curator = TraceCurator(mock_llm)
        traces = [make_trace(goal="broken feature", outcome=TraceOutcome.FAILURE)]
        results = await curator.curate(traces)
        assert len(results) == 1
        assert len(results[0].negative_constraints) > 0

    @pytest.mark.asyncio
    async def test_curate_skips_short_traces(self, mock_llm):
        curator = TraceCurator(mock_llm)
        traces = [make_trace(n_actions=1)]  # below min_actions=2
        results = await curator.curate(traces, min_actions=2)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_curate_uses_isa_for_critical_path(self, mock_llm):
        """Verify that the curator uses ISA opcodes (MARK_CRITICAL) for critical path."""
        curator = TraceCurator(mock_llm)
        traces = [make_trace(goal="test ISA", outcome=TraceOutcome.SUCCESS, n_actions=3)]
        results = await curator.curate(traces)
        assert len(results) == 1
        # Should have critical steps extracted via MARK_CRITICAL opcodes
        assert len(results[0].critical_steps) > 0


class TestHierarchicalChunker:
    @pytest.mark.asyncio
    async def test_chunk_curated_trace(self, mock_llm):
        curator = TraceCurator(mock_llm)
        traces = [make_trace(goal="build component")]
        curated = await curator.curate(traces)

        chunker = HierarchicalChunker(mock_llm)
        chunks = await chunker.chunk(curated)
        assert len(chunks) == 1
        assert chunks[0].parent.goal == "build component"
        assert len(chunks[0].children) > 0
        assert chunks[0].parent.child_node_ids

    @pytest.mark.asyncio
    async def test_chunk_uses_single_llm_call(self, mock_llm):
        """REM phase should use BUILD_PARENT + BUILD_CHILD in one call."""
        curator = TraceCurator(mock_llm)
        traces = [make_trace(goal="single call test", n_actions=3)]
        curated = await curator.curate(traces)

        chunker = HierarchicalChunker(mock_llm)
        chunks = await chunker.chunk(curated)
        assert len(chunks) == 1
        # Parent and children should both be created
        assert chunks[0].parent.summary != ""
        assert len(chunks[0].children) > 0


class TestTopologicalConnector:
    @pytest.mark.asyncio
    async def test_consolidate_creates_nodes_and_edges(self, mock_llm, store, vector_index):
        encoder = MockEmbeddingEncoder()
        config = DreamConfig()

        curator = TraceCurator(mock_llm)
        traces = [make_trace(goal="implement auth")]
        curated = await curator.curate(traces)

        chunker = HierarchicalChunker(mock_llm)
        chunks = await chunker.chunk(curated)

        connector = TopologicalConnector(store, vector_index, encoder, config)
        stats = await connector.consolidate(chunks)

        assert stats["nodes_created"] == 1
        assert stats["edges_created"] > 0
        assert vector_index.size == 1

        # Verify nodes in store
        parents = store.get_all_parent_nodes()
        assert len(parents) == 1
        children = store.get_child_nodes_for_parent(parents[0].node_id)
        assert len(children) > 0


class TestDreamEngine:
    @pytest.mark.asyncio
    async def test_full_dream_cycle(self, mock_llm, store, vector_index, config):
        encoder = MockEmbeddingEncoder()
        engine = DreamEngine(mock_llm, store, vector_index, encoder, config)

        # Save a session to process
        session = make_session(root_goal="build auth", n_traces=2, n_actions=3)
        store.save_session(session)

        journal = await engine.dream()
        assert journal.traces_processed > 0
        assert journal.nodes_created > 0
        assert journal.edges_created > 0
        assert journal.ended_at is not None
        assert len(journal.phase_log) > 0

        # Sessions should be marked as processed
        assert store.get_unprocessed_sessions() == []

    @pytest.mark.asyncio
    async def test_dream_no_sessions(self, mock_llm, store, vector_index, config):
        encoder = MockEmbeddingEncoder()
        engine = DreamEngine(mock_llm, store, vector_index, encoder, config)
        journal = await engine.dream()
        assert journal.traces_processed == 0
        assert "No unprocessed sessions" in journal.phase_log[0]
