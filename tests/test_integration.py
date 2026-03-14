"""Integration test: capture → dream → query → traverse."""

import pytest

from evolving_memory.config import CTEConfig
from evolving_memory.capture.session import SessionManager
from evolving_memory.capture.trace_logger import TraceLogger
from evolving_memory.dream.engine import DreamEngine
from evolving_memory.models.hierarchy import HierarchyLevel, RouterPath
from evolving_memory.router.cognitive_router import CognitiveRouter
from evolving_memory.storage.sqlite_store import SQLiteStore
from evolving_memory.storage.vector_index import VectorIndex

from conftest import MockLLMProvider, MockEmbeddingEncoder


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_cycle(self):
        """Capture a session, run dream cycle, query memory, traverse graph."""
        # Setup
        store = SQLiteStore(":memory:")
        index = VectorIndex(dim=384)
        encoder = MockEmbeddingEncoder()
        llm = MockLLMProvider()
        config = CTEConfig()

        # Step 1: Capture
        mgr = SessionManager(store)
        with mgr.session("build authentication system") as logger:
            with logger.trace(HierarchyLevel.GOAL, "implement auth") as ctx:
                ctx.action("analyze", "review requirements", result="OAuth + JWT needed")

            with logger.trace(HierarchyLevel.TACTICAL, "implement JWT") as ctx:
                ctx.action("research", "study JWT spec", result="understood claims, signing")
                ctx.action("code", "write jwt_utils.py", result="200 lines, encode/decode")
                ctx.action("test", "run pytest", result="5/5 passing")

            with logger.trace(HierarchyLevel.TACTICAL, "implement OAuth flow") as ctx:
                ctx.action("design", "sequence diagram", result="auth code flow designed")
                ctx.action("code", "write oauth.py", result="150 lines, full flow")
                ctx.action("test", "integration test", result="3/3 passing")

        # Verify capture
        sessions = store.get_unprocessed_sessions()
        assert len(sessions) == 1
        assert len(sessions[0].traces) == 3

        # Step 2: Dream (uses ISA opcodes internally)
        engine = DreamEngine(llm, store, index, encoder, config)
        journal = await engine.dream()
        assert journal.traces_processed > 0
        assert journal.nodes_created > 0

        # Sessions should be processed now
        assert store.get_unprocessed_sessions() == []

        # Step 3: Query
        router = CognitiveRouter(store, index, encoder, config.router)

        # Query for something we stored
        parents = store.get_all_parent_nodes()
        assert len(parents) > 0

        # Query with a parent's summary text (guaranteed to find it)
        decision = router.query(parents[0].summary)
        # With mock encoder, it should find the memory
        assert decision.path in (RouterPath.MEMORY_TRAVERSAL, RouterPath.ZERO_SHOT)

        # If we got a memory traversal, test the traversal
        if decision.path == RouterPath.MEMORY_TRAVERSAL:
            state = router.begin_traversal(decision.entry_point)
            steps = []
            while True:
                child, state = router.next_step(state)
                if child is None:
                    break
                steps.append(child)
            assert len(steps) > 0

        # Cleanup
        store.close()

    @pytest.mark.asyncio
    async def test_multiple_dream_cycles(self):
        """Test that multiple dream cycles accumulate knowledge."""
        store = SQLiteStore(":memory:")
        index = VectorIndex(dim=384)
        encoder = MockEmbeddingEncoder()
        llm = MockLLMProvider()
        config = CTEConfig()

        engine = DreamEngine(llm, store, index, encoder, config)
        mgr = SessionManager(store)

        # Cycle 1
        with mgr.session("session 1") as logger:
            with logger.trace(HierarchyLevel.TACTICAL, "task A") as ctx:
                ctx.action("do", "action A1", result="result A1")
                ctx.action("do", "action A2", result="result A2")

        j1 = await engine.dream()
        count_after_1 = len(store.get_all_parent_nodes())

        # Cycle 2
        with mgr.session("session 2") as logger:
            with logger.trace(HierarchyLevel.TACTICAL, "task B") as ctx:
                ctx.action("do", "action B1", result="result B1")
                ctx.action("do", "action B2", result="result B2")

        j2 = await engine.dream()
        count_after_2 = len(store.get_all_parent_nodes())

        # Should have accumulated nodes
        assert count_after_2 >= count_after_1
        assert j1.nodes_created > 0
        assert j2.nodes_created + j2.nodes_merged > 0

        store.close()
