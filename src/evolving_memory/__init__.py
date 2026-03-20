"""Evolving Memory — Cognitive Trajectory Engine for LLM agents."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from .capture.session import SessionManager
from .capture.trace_logger import TraceLogger
from .config import CTEConfig, ISAConfig
from .dream.engine import DreamEngine
from .embeddings.encoder import EmbeddingEncoder
from .isa.opcodes import ISA_VERSION, ISAVersionRegistry, Instruction, Opcode, Program
from .isa.parser import InstructionParser
from .isa.serializer import serialize_instruction, serialize_program
from .llm.base import BaseLLMProvider
from .models.graph import ChildNode, ParentNode
from .models.hierarchy import HierarchyLevel, RouterPath
from .models.query import EntryPoint, RouterDecision, TraversalState
from .models.strategy import DreamJournalEntry
from .router.cognitive_router import CognitiveRouter
from .storage.sqlite_store import SQLiteStore
from .storage.vector_index import VectorIndex
from .vm.context import VMContext, VMResult
from .vm.machine import CognitiveVM

__all__ = [
    # Engine
    "CognitiveTrajectoryEngine",
    "CTEConfig",
    "ISAConfig",
    # ISA
    "ISA_VERSION",
    "ISAVersionRegistry",
    "Opcode",
    "Instruction",
    "Program",
    "InstructionParser",
    "serialize_instruction",
    "serialize_program",
    # VM
    "CognitiveVM",
    "VMContext",
    "VMResult",
    # LLM
    "BaseLLMProvider",
    # Models
    "HierarchyLevel",
    "RouterPath",
    "RouterDecision",
    "TraversalState",
    "DreamJournalEntry",
    "ParentNode",
    "ChildNode",
    "EntryPoint",
]


class CognitiveTrajectoryEngine:
    """Public facade for the Cognitive Trajectory Engine.

    Usage::

        cte = CognitiveTrajectoryEngine(llm=my_provider)

        # Capture traces
        with cte.session("build auth") as logger:
            with logger.trace(HierarchyLevel.TACTICAL, "implement JWT") as ctx:
                ctx.action("write code", "Edit(auth.py)", result="done")

        # Dream (consolidate traces into memory graph)
        journal = await cte.dream()

        # Query memory
        decision = cte.query("how to implement JWT?")
        if decision.path == RouterPath.MEMORY_TRAVERSAL:
            state = cte.begin_traversal(decision.entry_point)
            while True:
                child, state = cte.next_step(state)
                if child is None:
                    break
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        dreaming_llm: BaseLLMProvider | None = None,
        config: CTEConfig | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self._config = config or CTEConfig()
        if db_path is not None:
            self._config.db_path = Path(db_path)

        self._store = SQLiteStore(self._config.db_path)
        self._encoder = EmbeddingEncoder(self._config.embedding_model, dim=self._config.embedding_dim)
        self._index = VectorIndex(
            dim=self._config.embedding_dim,
            index_path=self._config.faiss_path,
        )
        self._session_mgr = SessionManager(self._store)
        self._dream_engine = DreamEngine(
            llm=dreaming_llm or llm,
            store=self._store,
            index=self._index,
            encoder=self._encoder,
            config=self._config,
        )
        self._router = CognitiveRouter(
            store=self._store,
            index=self._index,
            encoder=self._encoder,
            config=self._config.router,
        )

    # ── Capture ─────────────────────────────────────────────────────

    @contextmanager
    def session(self, root_goal: str) -> Generator[TraceLogger, None, None]:
        """Open a trace capture session."""
        with self._session_mgr.session(root_goal) as logger:
            yield logger

    # ── Dream ───────────────────────────────────────────────────────

    async def dream(self) -> DreamJournalEntry:
        """Run a dream cycle to consolidate traces into the thought graph."""
        return await self._dream_engine.dream()

    # ── Query & Traverse ────────────────────────────────────────────

    def query(self, query_text: str) -> RouterDecision:
        """Route a query through the cognitive router."""
        return self._router.query(query_text)

    def begin_traversal(self, entry_point: EntryPoint) -> TraversalState:
        """Start step-by-step traversal of a memory entry point."""
        return self._router.begin_traversal(entry_point)

    def next_step(self, state: TraversalState) -> tuple[ChildNode | None, TraversalState]:
        """Load the next step in a traversal."""
        return self._router.next_step(state)

    def check_anomaly(self, state: TraversalState, current_context: str) -> TraversalState:
        """Check for semantic drift during traversal."""
        return self._router.check_anomaly(state, current_context)

    # ── Lifecycle ───────────────────────────────────────────────────

    def save_index(self) -> None:
        """Persist the FAISS index to disk."""
        self._index.save()

    def close(self) -> None:
        """Close the store connection."""
        self._store.close()
