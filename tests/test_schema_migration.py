"""Tests for ISA versioning, schema migrations, and cognitive migration during dream."""

import sqlite3

import pytest

from evolving_memory.isa.opcodes import (
    ISA_VERSION,
    ISAVersionRegistry,
    OPCODE_BY_NAME,
    get_registry,
)
from evolving_memory.isa.parser import InstructionParser
from evolving_memory.models.graph import ParentNode
from evolving_memory.models.hierarchy import HierarchyLevel, TraceOutcome
from evolving_memory.models.trace import TraceEntry
from evolving_memory.storage.migrations import run_migrations
from evolving_memory.storage.sqlite_store import SQLiteStore
from evolving_memory.dream.engine import DreamEngine
from evolving_memory.dream.migration import MigrationTransform
from evolving_memory.models.graph import ChildNode
from evolving_memory.config import CTEConfig

from conftest import (
    make_parent_node,
    make_session,
    MockEmbeddingEncoder,
    MockLLMProvider,
)


# ── ISAVersionRegistry ─────────────────────────────────────────────


class TestISAVersionRegistry:
    def test_register_and_lookup(self):
        reg = ISAVersionRegistry()
        reg.register("0.9", {"MEM_PTR", "HALT"})
        assert reg.get("0.9") == {"MEM_PTR", "HALT"}

    def test_unknown_version_returns_none(self):
        reg = ISAVersionRegistry()
        assert reg.get("999.0") is None

    def test_all_versions(self):
        reg = ISAVersionRegistry()
        reg.register("1.0", set())
        reg.register("0.9", set())
        assert reg.all_versions() == ["0.9", "1.0"]

    def test_current_returns_isa_version(self):
        reg = ISAVersionRegistry()
        assert reg.current() == ISA_VERSION

    def test_supports(self):
        reg = ISAVersionRegistry()
        reg.register("1.0", set())
        assert reg.supports("1.0")
        assert not reg.supports("2.0")

    def test_global_registry_has_v1(self):
        reg = get_registry()
        assert reg.supports(ISA_VERSION)
        opcodes = reg.get(ISA_VERSION)
        assert opcodes is not None
        assert "HALT" in opcodes
        assert "MEM_PTR" in opcodes


# ── Version-Aware Parser ───────────────────────────────────────────


class TestVersionAwareParser:
    def test_default_parser_stamps_current_version(self):
        parser = InstructionParser()
        program = parser.parse("HALT")
        assert program.isa_version == ISA_VERSION

    def test_explicit_version_stamps_program(self):
        parser = InstructionParser(isa_version="0.9")
        program = parser.parse("HALT")
        assert program.isa_version == "0.9"

    def test_legacy_fallback_resolves_opcode(self):
        """Even with unknown version, parser falls back to full opcode set."""
        parser = InstructionParser(isa_version="0.1")
        program = parser.parse("MEM_PTR some_query\nHALT")
        assert len(program.instructions) == 2
        assert program.parse_errors == []


# ── Schema Migrations ──────────────────────────────────────────────


class TestSchemaMigration:
    def test_fresh_db_runs_all_migrations(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # Create the base tables first (like SQLiteStore does)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS parent_nodes (
                node_id TEXT PRIMARY KEY,
                hierarchy_level INTEGER NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                access_count INTEGER NOT NULL DEFAULT 0,
                goal TEXT NOT NULL DEFAULT '',
                outcome TEXT NOT NULL DEFAULT 'unknown',
                trigger_goals TEXT NOT NULL DEFAULT '[]',
                negative_constraints TEXT NOT NULL DEFAULT '[]',
                child_node_ids TEXT NOT NULL DEFAULT '[]',
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                version INTEGER NOT NULL DEFAULT 1,
                domain TEXT NOT NULL DEFAULT 'default',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trace_entries (
                trace_id TEXT PRIMARY KEY,
                session_id TEXT,
                hierarchy_level INTEGER NOT NULL,
                parent_trace_id TEXT,
                goal TEXT NOT NULL DEFAULT '',
                outcome TEXT NOT NULL DEFAULT 'unknown',
                confidence REAL NOT NULL DEFAULT 0.0,
                source TEXT NOT NULL DEFAULT 'unknown_source',
                tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dream_journal (
                journal_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                traces_processed INTEGER NOT NULL DEFAULT 0,
                nodes_created INTEGER NOT NULL DEFAULT 0,
                nodes_merged INTEGER NOT NULL DEFAULT 0,
                edges_created INTEGER NOT NULL DEFAULT 0,
                constraints_extracted INTEGER NOT NULL DEFAULT 0,
                phase_log TEXT NOT NULL DEFAULT '[]'
            );
        """)
        count = run_migrations(conn)
        assert count == 2  # migration 1 (schema_version) + migration 2 (columns)

        # Verify schema_version table exists
        rows = conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
        assert [r[0] for r in rows] == [1, 2]

        # Verify isa_version columns exist
        conn.execute("SELECT isa_version FROM parent_nodes LIMIT 0")
        conn.execute("SELECT isa_version FROM trace_entries LIMIT 0")
        conn.execute("SELECT traces_migrated, nodes_migrated FROM dream_journal LIMIT 0")
        conn.close()

    def test_idempotent(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE parent_nodes (node_id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT '');
            CREATE TABLE trace_entries (trace_id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT '');
            CREATE TABLE dream_journal (journal_id TEXT PRIMARY KEY, started_at TEXT NOT NULL DEFAULT '');
        """)
        first = run_migrations(conn)
        second = run_migrations(conn)
        assert first == 2
        assert second == 0  # all already applied
        conn.close()

    def test_sqlite_store_creates_columns_on_fresh_db(self):
        """SQLiteStore on a brand-new DB should have isa_version columns."""
        store = SQLiteStore(":memory:")
        # These should not raise
        store._conn.execute("SELECT isa_version FROM parent_nodes LIMIT 0")
        store._conn.execute("SELECT isa_version FROM trace_entries LIMIT 0")
        store._conn.execute("SELECT traces_migrated, nodes_migrated FROM dream_journal LIMIT 0")
        store.close()


# ── Model ISA Version Defaults ────────────────────────────────────


class TestTraceEntryISAVersion:
    def test_default_isa_version(self):
        trace = TraceEntry(
            hierarchy_level=HierarchyLevel.TACTICAL,
            goal="test",
        )
        assert trace.isa_version == ISA_VERSION

    def test_store_roundtrip(self, store):
        session = make_session(root_goal="version test", n_traces=1, n_actions=2)
        store.save_session(session)
        sessions = store.get_unprocessed_sessions()
        assert len(sessions) == 1
        trace = sessions[0].traces[0]
        assert trace.isa_version == ISA_VERSION


class TestParentNodeISAVersion:
    def test_default_isa_version(self):
        node = make_parent_node()
        assert node.isa_version == ISA_VERSION

    def test_store_roundtrip(self, store):
        node = make_parent_node(goal="version roundtrip")
        store.save_parent_node(node)
        retrieved = store.get_parent_node(node.node_id)
        assert retrieved.isa_version == ISA_VERSION


# ── Legacy Data Helpers ────────────────────────────────────────────


class TestLegacyHelpers:
    def test_get_legacy_parent_nodes(self, store):
        # Save a node, manually set its isa_version to old
        node = make_parent_node(goal="old strategy")
        store.save_parent_node(node)
        store._conn.execute(
            "UPDATE parent_nodes SET isa_version = '0.9' WHERE node_id = ?",
            (node.node_id,),
        )
        store._conn.commit()

        legacy = store.get_legacy_parent_nodes()
        assert len(legacy) == 1
        assert legacy[0].isa_version == "0.9"

    def test_get_legacy_trace_count(self, store):
        session = make_session(n_traces=2, n_actions=2)
        store.save_session(session)
        # Mark one trace as legacy
        trace_id = session.traces[0].trace_id
        store._conn.execute(
            "UPDATE trace_entries SET isa_version = '0.9' WHERE trace_id = ?",
            (trace_id,),
        )
        store._conn.commit()

        assert store.get_legacy_trace_count() == 1

    def test_update_parent_node_isa_version(self, store):
        node = make_parent_node()
        store.save_parent_node(node)
        store.update_parent_node_isa_version(node.node_id, "2.0")
        updated = store.get_parent_node(node.node_id)
        assert updated.isa_version == "2.0"


# ── Cognitive Migration During Dream ──────────────────────────────


class TestCognitiveMigration:
    @pytest.mark.asyncio
    async def test_dream_migrates_legacy_nodes(self, store, vector_index, config):
        """Dream cycle Phase 0 should re-stamp legacy nodes."""
        llm = MockLLMProvider()
        encoder = MockEmbeddingEncoder()
        engine = DreamEngine(llm, store, vector_index, encoder, config)

        # Create a parent node with old ISA version
        node = make_parent_node(goal="legacy strategy")
        store.save_parent_node(node)
        store._conn.execute(
            "UPDATE parent_nodes SET isa_version = '0.9' WHERE node_id = ?",
            (node.node_id,),
        )
        store._conn.commit()

        # Verify it's legacy
        assert len(store.get_legacy_parent_nodes()) == 1

        # Run dream (no sessions to process, but migration still runs)
        journal = await engine.dream()

        # Node should be migrated
        assert store.get_legacy_parent_nodes() == []
        updated = store.get_parent_node(node.node_id)
        assert updated.isa_version == ISA_VERSION

        # Journal should record migration
        assert journal.nodes_migrated == 1
        assert any("Phase 0" in entry for entry in journal.phase_log)

    @pytest.mark.asyncio
    async def test_dream_migrates_legacy_traces(self, store, vector_index, config):
        """Dream cycle Phase 0 should re-stamp legacy traces."""
        llm = MockLLMProvider()
        encoder = MockEmbeddingEncoder()
        engine = DreamEngine(llm, store, vector_index, encoder, config)

        # Create a session with traces, then mark them as legacy
        session = make_session(n_traces=3, n_actions=2)
        store.save_session(session)
        store.mark_session_processed(session.session_id)  # so dream doesn't process them
        for trace in session.traces:
            store._conn.execute(
                "UPDATE trace_entries SET isa_version = '0.8' WHERE trace_id = ?",
                (trace.trace_id,),
            )
        store._conn.commit()

        assert store.get_legacy_trace_count() == 3

        journal = await engine.dream()

        assert store.get_legacy_trace_count() == 0
        assert journal.traces_migrated == 3

    @pytest.mark.asyncio
    async def test_dream_no_legacy_skips_phase0(self, store, vector_index, config):
        """If no legacy data exists, Phase 0 should be silent."""
        llm = MockLLMProvider()
        encoder = MockEmbeddingEncoder()
        engine = DreamEngine(llm, store, vector_index, encoder, config)

        journal = await engine.dream()
        assert journal.nodes_migrated == 0
        assert journal.traces_migrated == 0
        assert not any("Phase 0" in entry for entry in journal.phase_log)

    @pytest.mark.asyncio
    async def test_migration_transform_enriches_nodes(self, store, vector_index, config):
        """MigrationTransform should enrich legacy nodes during Phase 0."""

        class MockTransform(MigrationTransform):
            from_version = "0.9"
            to_version = "1.0"

            async def transform(self, node, children, llm):
                node.content = f"{node.content}\n[enriched: risk_level=high]"
                node.summary = f"[RISK: HIGH] {node.summary}"
                for child in children:
                    child.content = f"{child.content}\n[risk_level: medium]"
                return node, children

        llm = MockLLMProvider()
        encoder = MockEmbeddingEncoder()
        engine = DreamEngine(llm, store, vector_index, encoder, config)
        engine.register_migration(MockTransform())

        # Create a parent node with children at old ISA version
        node = make_parent_node(goal="legacy strategy", summary="old summary")
        store.save_parent_node(node)
        child = ChildNode(
            parent_node_id=node.node_id,
            hierarchy_level=HierarchyLevel.TACTICAL,
            step_index=0,
            reasoning="step 1",
            action="do something",
            content="step content",
        )
        store.save_child_node(child)

        # Mark as legacy
        store._conn.execute(
            "UPDATE parent_nodes SET isa_version = '0.9' WHERE node_id = ?",
            (node.node_id,),
        )
        store._conn.commit()

        journal = await engine.dream()

        # Node should be migrated AND enriched
        updated = store.get_parent_node(node.node_id)
        assert updated.isa_version == ISA_VERSION
        assert "[enriched: risk_level=high]" in updated.content
        assert "[RISK: HIGH]" in updated.summary

        # Child should be enriched
        children = store.get_child_nodes_for_parent(node.node_id)
        assert len(children) == 1
        assert "[risk_level: medium]" in children[0].content

        # Journal should record enrichment
        assert journal.nodes_migrated == 1
        assert any("enriched" in entry for entry in journal.phase_log)

    @pytest.mark.asyncio
    async def test_migration_transform_version_mismatch_skips(self, store, vector_index, config):
        """Transform should only apply when from_version matches."""

        class WrongVersionTransform(MigrationTransform):
            from_version = "0.5"  # Doesn't match "0.9"
            to_version = "1.0"

            async def transform(self, node, children, llm):
                node.content = "SHOULD NOT HAPPEN"
                return node, children

        llm = MockLLMProvider()
        encoder = MockEmbeddingEncoder()
        engine = DreamEngine(llm, store, vector_index, encoder, config)
        engine.register_migration(WrongVersionTransform())

        node = make_parent_node(goal="test")
        original_content = node.content
        store.save_parent_node(node)
        store._conn.execute(
            "UPDATE parent_nodes SET isa_version = '0.9' WHERE node_id = ?",
            (node.node_id,),
        )
        store._conn.commit()

        await engine.dream()

        updated = store.get_parent_node(node.node_id)
        assert updated.content == original_content  # Not transformed
