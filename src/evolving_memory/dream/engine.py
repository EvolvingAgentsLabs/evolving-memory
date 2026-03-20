"""DreamEngine orchestrator — runs the 3-phase dream cycle (SWS → REM → Consolidation)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..config import CTEConfig
from ..embeddings.encoder import EmbeddingEncoder
from ..isa.opcodes import ISA_VERSION
from ..llm.base import BaseLLMProvider
from ..models.strategy import DreamJournalEntry
from ..storage.sqlite_store import SQLiteStore
from ..storage.vector_index import VectorIndex
from .adapters.default_adapter import DefaultAdapter
from .chunker import HierarchicalChunker
from .connector import TopologicalConnector
from .curator import TraceCurator
from .domain_adapter import DreamDomainAdapter

logger = logging.getLogger(__name__)


class DreamEngine:
    """Orchestrates the 3-phase dream cycle over unprocessed trace sessions."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        store: SQLiteStore,
        index: VectorIndex,
        encoder: EmbeddingEncoder,
        config: CTEConfig,
        adapter: DreamDomainAdapter | None = None,
    ) -> None:
        self._llm = llm
        self._store = store
        self._index = index
        self._encoder = encoder
        self._config = config
        self._adapter = adapter or DefaultAdapter()
        self._curator = TraceCurator(llm)
        self._chunker = HierarchicalChunker(llm)
        self._connector = TopologicalConnector(store, index, encoder, config.dream, llm)

    async def dream(self) -> DreamJournalEntry:
        """Run a full dream cycle over all unprocessed sessions."""
        journal = DreamJournalEntry()

        # Phase 0: Migrate legacy data to current ISA version
        self._migrate_legacy_data(journal)

        sessions = self._store.get_unprocessed_sessions()
        if not sessions:
            journal.phase_log.append("No unprocessed sessions found")
            journal.ended_at = datetime.now(timezone.utc)
            return journal

        # Collect all traces
        all_traces = []
        for session in sessions:
            all_traces.extend(session.traces)

        # Limit traces per cycle
        traces = all_traces[: self._config.dream.max_traces_per_cycle]
        journal.traces_processed = len(traces)

        # Phase 1: SWS — curate traces (ISA: EXTRACT_CONSTRAINT, MARK_CRITICAL)
        journal.phase_log.append(f"SWS: curating {len(traces)} traces")
        curated = await self._curator.curate(
            traces, min_actions=self._config.dream.min_actions_for_trace
        )
        journal.phase_log.append(f"SWS: {len(curated)} traces curated")

        total_constraints = sum(len(c.negative_constraints) for c in curated)
        journal.constraints_extracted = total_constraints

        # Phase 2: REM — create hierarchical nodes (ISA: BUILD_PARENT, BUILD_CHILD)
        journal.phase_log.append(f"REM: chunking {len(curated)} curated traces")
        chunks = await self._chunker.chunk(curated)
        journal.phase_log.append(f"REM: {len(chunks)} chunks created")

        # Phase 3: Consolidation — edges, embeddings, merge (algorithmic)
        journal.phase_log.append("Consolidation: connecting nodes")
        stats = await self._connector.consolidate(chunks)
        journal.nodes_created = stats["nodes_created"]
        journal.nodes_merged = stats["nodes_merged"]
        journal.edges_created = stats["edges_created"]
        journal.cross_edges_created = stats.get("cross_edges_created", 0)
        journal.phase_log.append(
            f"Consolidation: {stats['nodes_created']} created, "
            f"{stats['nodes_merged']} merged, {stats['edges_created']} edges"
            f" ({journal.cross_edges_created} cross-trace)"
        )

        # Mark sessions as processed
        for session in sessions:
            self._store.mark_session_processed(session.session_id)

        journal.ended_at = datetime.now(timezone.utc)
        self._store.save_journal_entry(journal)
        return journal

    def _migrate_legacy_data(self, journal: DreamJournalEntry) -> None:
        """Phase 0 — re-stamp legacy nodes/traces to current ISA version.

        This runs during every dream cycle as reconsolidation. Currently it
        just updates the isa_version field; future versions can apply
        structural translations (e.g. opcode renames, field transforms).
        """
        legacy_nodes = self._store.get_legacy_parent_nodes(ISA_VERSION)
        if legacy_nodes:
            for node in legacy_nodes:
                self._store.update_parent_node_isa_version(node.node_id, ISA_VERSION)
            journal.nodes_migrated = len(legacy_nodes)
            journal.phase_log.append(
                f"Phase 0: migrated {len(legacy_nodes)} legacy parent nodes to ISA {ISA_VERSION}"
            )
            logger.info("Migrated %d legacy parent nodes to ISA %s", len(legacy_nodes), ISA_VERSION)

        legacy_trace_count = self._store.get_legacy_trace_count(ISA_VERSION)
        if legacy_trace_count > 0:
            # Bulk update traces
            self._store._conn.execute(
                "UPDATE trace_entries SET isa_version = ? WHERE isa_version != ?",
                (ISA_VERSION, ISA_VERSION),
            )
            self._store._conn.commit()
            journal.traces_migrated = legacy_trace_count
            journal.phase_log.append(
                f"Phase 0: migrated {legacy_trace_count} legacy traces to ISA {ISA_VERSION}"
            )
            logger.info("Migrated %d legacy traces to ISA %s", legacy_trace_count, ISA_VERSION)
