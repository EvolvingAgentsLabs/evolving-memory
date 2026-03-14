"""SessionManager — wraps session lifecycle and persistence."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from ..models.trace import TraceSession
from ..storage.sqlite_store import SQLiteStore
from .trace_logger import TraceLogger


class SessionManager:
    """Manages trace session lifecycle — creates, closes, and persists sessions."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    @contextmanager
    def session(self, root_goal: str) -> Generator[TraceLogger, None, None]:
        sess = TraceSession(root_goal=root_goal)
        logger = TraceLogger(session_id=sess.session_id)
        try:
            yield logger
        finally:
            sess.ended_at = datetime.now(timezone.utc)
            sess.traces = logger.traces
            self._store.save_session(sess)
