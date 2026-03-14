"""SQLite persistence for the thought graph, traces, and dream journal."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..models.graph import ParentNode, ChildNode, ThoughtEdge
from ..models.hierarchy import HierarchyLevel, TraceOutcome, TraceSource, EdgeType
from ..models.trace import TraceEntry, TraceSession, ActionEntry
from ..models.strategy import NegativeConstraint, DreamJournalEntry


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteStore:
    """Full CRUD for all CTE tables."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    # ── schema ──────────────────────────────────────────────────────

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
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

            CREATE INDEX IF NOT EXISTS idx_parent_domain ON parent_nodes(domain);

            CREATE TABLE IF NOT EXISTS child_nodes (
                node_id TEXT PRIMARY KEY,
                parent_node_id TEXT NOT NULL,
                hierarchy_level INTEGER NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                access_count INTEGER NOT NULL DEFAULT 0,
                step_index INTEGER NOT NULL DEFAULT 0,
                reasoning TEXT NOT NULL DEFAULT '',
                action TEXT NOT NULL DEFAULT '',
                result TEXT NOT NULL DEFAULT '',
                is_critical_path INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (parent_node_id) REFERENCES parent_nodes(node_id)
            );

            CREATE INDEX IF NOT EXISTS idx_child_parent ON child_nodes(parent_node_id);

            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source_node_id);
            CREATE INDEX IF NOT EXISTS idx_edge_target ON edges(target_node_id);

            CREATE TABLE IF NOT EXISTS trace_sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                root_goal TEXT NOT NULL DEFAULT '',
                processed INTEGER NOT NULL DEFAULT 0
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
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES trace_sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_trace_session ON trace_entries(session_id);

            CREATE TABLE IF NOT EXISTS action_entries (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                reasoning TEXT NOT NULL DEFAULT '',
                action_payload TEXT NOT NULL DEFAULT '',
                result TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (trace_id) REFERENCES trace_entries(trace_id)
            );

            CREATE INDEX IF NOT EXISTS idx_action_trace ON action_entries(trace_id);

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

            CREATE TABLE IF NOT EXISTS negative_constraints (
                constraint_id TEXT PRIMARY KEY,
                parent_node_id TEXT NOT NULL,
                description TEXT NOT NULL,
                source_trace_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (parent_node_id) REFERENCES parent_nodes(node_id)
            );
        """)
        self._conn.commit()

    # ── parent nodes ────────────────────────────────────────────────

    def save_parent_node(self, node: ParentNode, domain: str = "default") -> None:
        now = _now_iso()
        self._conn.execute(
            """INSERT OR REPLACE INTO parent_nodes
               (node_id, hierarchy_level, content, summary, confidence, access_count,
                goal, outcome, trigger_goals, negative_constraints, child_node_ids,
                success_count, failure_count, version, domain, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                node.node_id, int(node.hierarchy_level), node.content, node.summary,
                node.confidence, node.access_count, node.goal, node.outcome.value,
                json.dumps(node.trigger_goals), json.dumps(node.negative_constraints),
                json.dumps(node.child_node_ids), node.success_count, node.failure_count,
                node.version, domain, node.created_at.isoformat(), now,
            ),
        )
        self._conn.commit()

    def get_parent_node(self, node_id: str) -> ParentNode | None:
        row = self._conn.execute(
            "SELECT * FROM parent_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_parent(row)

    def get_all_parent_nodes(self, domain: str | None = None) -> list[ParentNode]:
        if domain:
            rows = self._conn.execute(
                "SELECT * FROM parent_nodes WHERE domain = ?", (domain,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM parent_nodes").fetchall()
        return [self._row_to_parent(r) for r in rows]

    def get_domains(self) -> list[str]:
        """Return all distinct domains."""
        rows = self._conn.execute(
            "SELECT DISTINCT domain FROM parent_nodes ORDER BY domain"
        ).fetchall()
        return [r["domain"] for r in rows]

    def get_stats(self) -> dict:
        """Return summary statistics about the store."""
        parent_count = self._conn.execute("SELECT COUNT(*) FROM parent_nodes").fetchone()[0]
        child_count = self._conn.execute("SELECT COUNT(*) FROM child_nodes").fetchone()[0]
        edge_count = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        session_count = self._conn.execute("SELECT COUNT(*) FROM trace_sessions").fetchone()[0]
        trace_count = self._conn.execute("SELECT COUNT(*) FROM trace_entries").fetchone()[0]
        journal_count = self._conn.execute("SELECT COUNT(*) FROM dream_journal").fetchone()[0]
        return {
            "parent_nodes": parent_count,
            "child_nodes": child_count,
            "edges": edge_count,
            "sessions": session_count,
            "traces": trace_count,
            "dream_cycles": journal_count,
        }

    def increment_access(self, node_id: str) -> None:
        self._conn.execute(
            "UPDATE parent_nodes SET access_count = access_count + 1, updated_at = ? WHERE node_id = ?",
            (_now_iso(), node_id),
        )
        self._conn.commit()

    def _row_to_parent(self, row: sqlite3.Row) -> ParentNode:
        return ParentNode(
            node_id=row["node_id"],
            hierarchy_level=HierarchyLevel(row["hierarchy_level"]),
            content=row["content"],
            summary=row["summary"],
            confidence=row["confidence"],
            access_count=row["access_count"],
            goal=row["goal"],
            outcome=TraceOutcome(row["outcome"]),
            trigger_goals=json.loads(row["trigger_goals"]),
            negative_constraints=json.loads(row["negative_constraints"]),
            child_node_ids=json.loads(row["child_node_ids"]),
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            version=row["version"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # ── child nodes ─────────────────────────────────────────────────

    def save_child_node(self, node: ChildNode) -> None:
        now = _now_iso()
        self._conn.execute(
            """INSERT OR REPLACE INTO child_nodes
               (node_id, parent_node_id, hierarchy_level, content, summary, confidence,
                access_count, step_index, reasoning, action, result, is_critical_path,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                node.node_id, node.parent_node_id, int(node.hierarchy_level),
                node.content, node.summary, node.confidence, node.access_count,
                node.step_index, node.reasoning, node.action, node.result,
                int(node.is_critical_path), node.created_at.isoformat(), now,
            ),
        )
        self._conn.commit()

    def get_child_nodes_for_parent(self, parent_node_id: str) -> list[ChildNode]:
        rows = self._conn.execute(
            "SELECT * FROM child_nodes WHERE parent_node_id = ? ORDER BY step_index",
            (parent_node_id,),
        ).fetchall()
        return [self._row_to_child(r) for r in rows]

    def get_child_node(self, node_id: str) -> ChildNode | None:
        row = self._conn.execute(
            "SELECT * FROM child_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_child(row)

    def _row_to_child(self, row: sqlite3.Row) -> ChildNode:
        return ChildNode(
            node_id=row["node_id"],
            parent_node_id=row["parent_node_id"],
            hierarchy_level=HierarchyLevel(row["hierarchy_level"]),
            content=row["content"],
            summary=row["summary"],
            confidence=row["confidence"],
            access_count=row["access_count"],
            step_index=row["step_index"],
            reasoning=row["reasoning"],
            action=row["action"],
            result=row["result"],
            is_critical_path=bool(row["is_critical_path"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # ── edges ───────────────────────────────────────────────────────

    def save_edge(self, edge: ThoughtEdge) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO edges
               (edge_id, source_node_id, target_node_id, edge_type, weight, created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                edge.edge_id, edge.source_node_id, edge.target_node_id,
                edge.edge_type.value, edge.weight, edge.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_edges_from(self, node_id: str) -> list[ThoughtEdge]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE source_node_id = ?", (node_id,)
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_edges_to(self, node_id: str) -> list[ThoughtEdge]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE target_node_id = ?", (node_id,)
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def _row_to_edge(self, row: sqlite3.Row) -> ThoughtEdge:
        return ThoughtEdge(
            edge_id=row["edge_id"],
            source_node_id=row["source_node_id"],
            target_node_id=row["target_node_id"],
            edge_type=EdgeType(row["edge_type"]),
            weight=row["weight"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ── trace sessions ──────────────────────────────────────────────

    def save_session(self, session: TraceSession) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO trace_sessions
               (session_id, started_at, ended_at, root_goal)
               VALUES (?,?,?,?)""",
            (
                session.session_id, session.started_at.isoformat(),
                session.ended_at.isoformat() if session.ended_at else None,
                session.root_goal,
            ),
        )
        for trace in session.traces:
            self.save_trace(trace)
        self._conn.commit()

    def save_trace(self, trace: TraceEntry) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO trace_entries
               (trace_id, session_id, hierarchy_level, parent_trace_id, goal,
                outcome, confidence, source, tags, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                trace.trace_id, trace.session_id, int(trace.hierarchy_level),
                trace.parent_trace_id, trace.goal, trace.outcome.value,
                trace.confidence, trace.source.value,
                json.dumps(trace.tags), trace.created_at.isoformat(),
            ),
        )
        for action in trace.action_entries:
            self._conn.execute(
                """INSERT INTO action_entries (trace_id, timestamp, reasoning, action_payload, result)
                   VALUES (?,?,?,?,?)""",
                (
                    trace.trace_id, action.timestamp.isoformat(),
                    action.reasoning, action.action_payload, action.result,
                ),
            )
        self._conn.commit()

    def get_unprocessed_sessions(self) -> list[TraceSession]:
        rows = self._conn.execute(
            "SELECT * FROM trace_sessions WHERE processed = 0 ORDER BY started_at"
        ).fetchall()
        sessions = []
        for row in rows:
            session = TraceSession(
                session_id=row["session_id"],
                started_at=datetime.fromisoformat(row["started_at"]),
                ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
                root_goal=row["root_goal"],
            )
            session.traces = self._get_traces_for_session(session.session_id)
            sessions.append(session)
        return sessions

    def mark_session_processed(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE trace_sessions SET processed = 1 WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()

    def _get_traces_for_session(self, session_id: str) -> list[TraceEntry]:
        rows = self._conn.execute(
            "SELECT * FROM trace_entries WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        traces = []
        for row in rows:
            actions = self._get_actions_for_trace(row["trace_id"])
            traces.append(TraceEntry(
                trace_id=row["trace_id"],
                session_id=row["session_id"],
                hierarchy_level=HierarchyLevel(row["hierarchy_level"]),
                parent_trace_id=row["parent_trace_id"],
                goal=row["goal"],
                outcome=TraceOutcome(row["outcome"]),
                confidence=row["confidence"],
                source=TraceSource(row["source"]) if row["source"] else TraceSource.UNKNOWN_SOURCE,
                action_entries=actions,
                tags=json.loads(row["tags"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            ))
        return traces

    def _get_actions_for_trace(self, trace_id: str) -> list[ActionEntry]:
        rows = self._conn.execute(
            "SELECT * FROM action_entries WHERE trace_id = ? ORDER BY rowid",
            (trace_id,),
        ).fetchall()
        return [
            ActionEntry(
                timestamp=datetime.fromisoformat(r["timestamp"]),
                reasoning=r["reasoning"],
                action_payload=r["action_payload"],
                result=r["result"],
            )
            for r in rows
        ]

    # ── dream journal ───────────────────────────────────────────────

    def save_journal_entry(self, entry: DreamJournalEntry) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO dream_journal
               (journal_id, started_at, ended_at, traces_processed,
                nodes_created, nodes_merged, edges_created, constraints_extracted, phase_log)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                entry.journal_id, entry.started_at.isoformat(),
                entry.ended_at.isoformat() if entry.ended_at else None,
                entry.traces_processed, entry.nodes_created, entry.nodes_merged,
                entry.edges_created, entry.constraints_extracted,
                json.dumps(entry.phase_log),
            ),
        )
        self._conn.commit()

    # ── negative constraints ────────────────────────────────────────

    def save_negative_constraint(self, constraint: NegativeConstraint) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO negative_constraints
               (constraint_id, parent_node_id, description, source_trace_id, created_at)
               VALUES (?,?,?,?,?)""",
            (
                constraint.constraint_id, constraint.parent_node_id,
                constraint.description, constraint.source_trace_id,
                constraint.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_constraints_for_parent(self, parent_node_id: str) -> list[NegativeConstraint]:
        rows = self._conn.execute(
            "SELECT * FROM negative_constraints WHERE parent_node_id = ?",
            (parent_node_id,),
        ).fetchall()
        return [
            NegativeConstraint(
                constraint_id=r["constraint_id"],
                parent_node_id=r["parent_node_id"],
                description=r["description"],
                source_trace_id=r["source_trace_id"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    # ── lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()
