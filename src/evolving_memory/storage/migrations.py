"""SQLite schema migration system — additive, idempotent migrations."""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

# Each migration is (version_number, description, SQL).
# Migrations are applied in order and tracked in `schema_version`.
MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "Create schema_version table",
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """,
    ),
    (
        2,
        "Add isa_version columns and migration stats",
        """
        ALTER TABLE parent_nodes ADD COLUMN isa_version TEXT NOT NULL DEFAULT '1.0';
        ALTER TABLE trace_entries ADD COLUMN isa_version TEXT NOT NULL DEFAULT '1.0';
        ALTER TABLE dream_journal ADD COLUMN traces_migrated INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE dream_journal ADD COLUMN nodes_migrated INTEGER NOT NULL DEFAULT 0;
        """,
    ),
    (
        3,
        "Add failure_class columns",
        """
        ALTER TABLE trace_entries ADD COLUMN failure_class TEXT NOT NULL DEFAULT '';
        ALTER TABLE negative_constraints ADD COLUMN failure_class TEXT NOT NULL DEFAULT '';
        """,
    ),
]


def _get_applied_versions(conn: sqlite3.Connection) -> set[int]:
    """Return the set of already-applied migration versions."""
    try:
        rows = conn.execute("SELECT version FROM schema_version").fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        # schema_version table doesn't exist yet — no migrations applied
        return set()


def run_migrations(conn: sqlite3.Connection) -> int:
    """Run all pending migrations. Returns count of migrations applied."""
    applied = _get_applied_versions(conn)
    count = 0

    for version, description, sql in MIGRATIONS:
        if version in applied:
            continue

        logger.info("Applying migration %d: %s", version, description)
        # Execute each statement separately (ALTER TABLE can't be in executescript
        # with other ALTERs in some SQLite builds)
        for statement in sql.strip().split(";"):
            statement = statement.strip()
            if statement:
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError as exc:
                    # Tolerate "duplicate column" errors for idempotency
                    if "duplicate column" in str(exc).lower():
                        logger.debug("Column already exists, skipping: %s", exc)
                    else:
                        raise

        # Record that this migration was applied
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, description) VALUES (?, ?)",
            (version, description),
        )
        conn.commit()
        count += 1

    return count
