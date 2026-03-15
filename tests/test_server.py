"""Tests for the FastAPI server endpoints."""

from __future__ import annotations

import pytest

from conftest import MockLLMProvider, MockEmbeddingEncoder

# Skip all tests if fastapi/httpx not installed
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from httpx import AsyncClient, ASGITransport
from evolving_memory.config import CTEConfig
from evolving_memory.server.app import MemoryServer, create_app
from evolving_memory.storage.sqlite_store import SQLiteStore
from evolving_memory.storage.vector_index import VectorIndex


@pytest.fixture
def server():
    """Create a test server with mock components."""
    llm = MockLLMProvider()
    config = CTEConfig()
    srv = MemoryServer.__new__(MemoryServer)
    srv.config = config
    srv.store = SQLiteStore(":memory:")
    srv.encoder = MockEmbeddingEncoder(dim=768)
    srv.index = VectorIndex(dim=768)
    from evolving_memory.router.cognitive_router import CognitiveRouter
    srv.router = CognitiveRouter(
        store=srv.store, index=srv.index,
        encoder=srv.encoder, config=config.router,
    )
    from evolving_memory.dream.engine import DreamEngine
    srv._engines = {
        "default": DreamEngine(
            llm=llm, store=srv.store, index=srv.index,
            encoder=srv.encoder, config=config,
        )
    }
    srv.get_engine = lambda domain="default": srv._engines.get(domain, srv._engines["default"])
    yield srv
    srv.store.close()


@pytest.fixture
def app(server):
    return create_app(server)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealth:
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestTraceIngestion:
    async def test_ingest_trace(self, client):
        resp = await client.post("/traces", json={
            "goal": "navigate to door",
            "hierarchy_level": 3,
            "outcome": "success",
            "confidence": 0.9,
            "source": "real_world",
            "actions": [
                {"reasoning": "see door", "action_payload": "MOVE_FORWARD", "result": "moved"},
                {"reasoning": "at door", "action_payload": "STOP", "result": "stopped"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "trace_id" in data
        assert "session_id" in data

    async def test_ingest_minimal_trace(self, client):
        resp = await client.post("/traces", json={"goal": "test"})
        assert resp.status_code == 200


class TestDreamCycle:
    async def test_dream_no_traces(self, client):
        resp = await client.post("/dream/run", json={"domain": "default"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["traces_processed"] == 0

    async def test_dream_with_traces(self, client):
        # Ingest a trace first
        await client.post("/traces", json={
            "goal": "test goal",
            "outcome": "success",
            "confidence": 0.8,
            "actions": [
                {"reasoning": "step 1", "action_payload": "act 1", "result": "ok"},
                {"reasoning": "step 2", "action_payload": "act 2", "result": "ok"},
                {"reasoning": "step 3", "action_payload": "act 3", "result": "ok"},
            ],
        })
        resp = await client.post("/dream/run")
        assert resp.status_code == 200
        data = resp.json()
        assert "journal_id" in data


class TestQuery:
    async def test_query_empty(self, client):
        resp = await client.get("/query", params={"q": "how to navigate"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == "zero_shot"


class TestNodeAccess:
    async def test_get_nonexistent_node(self, client):
        resp = await client.get("/nodes/nonexistent")
        assert resp.status_code == 404

    async def test_get_children_empty(self, client):
        resp = await client.get("/nodes/nonexistent/children")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_traverse_empty(self, client):
        resp = await client.get("/nodes/nonexistent/traverse")
        assert resp.status_code == 200
        data = resp.json()
        assert data["edges_out"] == []
        assert data["edges_in"] == []


class TestRouter:
    async def test_route_empty(self, client):
        resp = await client.post("/route", json={"query": "test"})
        assert resp.status_code == 200
        assert resp.json()["path"] == "zero_shot"


class TestDomains:
    async def test_list_domains_empty(self, client):
        resp = await client.get("/domains")
        assert resp.status_code == 200
        assert resp.json() == {"domains": []}


class TestStats:
    async def test_stats_empty(self, client):
        resp = await client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["parent_nodes"] == 0
        assert data["sessions"] == 0

    async def test_stats_after_ingest(self, client):
        await client.post("/traces", json={
            "goal": "test",
            "actions": [
                {"reasoning": "r", "action_payload": "a", "result": "ok"},
            ],
        })
        resp = await client.get("/stats")
        data = resp.json()
        assert data["sessions"] == 1
        assert data["traces"] == 1
