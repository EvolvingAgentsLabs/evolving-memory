"""API routes for the evolving-memory server."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ..isa.opcodes import ISA_VERSION, get_registry
from ..models.hierarchy import HierarchyLevel, TraceOutcome, TraceSource
from ..models.trace import ActionEntry, TraceEntry, TraceSession
from ..capture.session import SessionManager

if TYPE_CHECKING:
    from .app import MemoryServer

logger = logging.getLogger(__name__)


# ── Request/Response Models ──────────────────────────────────────────


class TraceRequest(BaseModel):
    """Ingest a trace entry."""
    goal: str
    hierarchy_level: int = 3
    outcome: str = "unknown"
    confidence: float = 0.0
    source: str = "unknown_source"
    actions: list[dict] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class DreamRequest(BaseModel):
    """Trigger a dream cycle."""
    domain: str = "default"


class RouteRequest(BaseModel):
    """Router query."""
    query: str
    top_k: int = 5


# ── Router Factory ───────────────────────────────────────────────────


def create_router(server: "MemoryServer") -> APIRouter:
    router = APIRouter()

    # ── Health ────────────────────────────────────────────────────

    @router.get("/health")
    async def health():
        return {"status": "ok"}

    # ── Trace Ingestion ──────────────────────────────────────────

    @router.post("/traces")
    async def ingest_trace(req: TraceRequest):
        actions = [
            ActionEntry(
                reasoning=a.get("reasoning", ""),
                action_payload=a.get("action_payload", a.get("actionPayload", "")),
                result=a.get("result", ""),
            )
            for a in req.actions
        ]
        trace = TraceEntry(
            hierarchy_level=HierarchyLevel(req.hierarchy_level),
            goal=req.goal,
            outcome=TraceOutcome(req.outcome),
            confidence=req.confidence,
            source=TraceSource(req.source),
            action_entries=actions,
            tags=req.tags,
        )
        # Wrap in a session and save
        session = TraceSession(root_goal=req.goal)
        session.traces.append(trace)
        session.ended_at = datetime.now(timezone.utc)
        server.store.save_session(session)
        return {
            "trace_id": trace.trace_id,
            "session_id": session.session_id,
        }

    # ── Dream Cycle ──────────────────────────────────────────────

    @router.post("/dream/run")
    async def run_dream(req: DreamRequest | None = None):
        domain = req.domain if req else "default"
        engine = server.get_engine(domain)
        journal = await engine.dream()
        return {
            "journal_id": journal.journal_id,
            "traces_processed": journal.traces_processed,
            "nodes_created": journal.nodes_created,
            "nodes_merged": journal.nodes_merged,
            "edges_created": journal.edges_created,
            "constraints_extracted": journal.constraints_extracted,
            "phase_log": journal.phase_log,
        }

    # ── Query ────────────────────────────────────────────────────

    @router.get("/query")
    async def query(q: str = Query(..., description="Semantic search query")):
        decision = server.router.query(q)
        result = {
            "path": decision.path.value,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
        }
        if decision.entry_point:
            ep = decision.entry_point
            result["entry_point"] = {
                "node_id": ep.parent_node.node_id,
                "goal": ep.parent_node.goal,
                "summary": ep.parent_node.summary,
                "similarity_score": ep.similarity_score,
                "composite_score": ep.composite_score,
            }
        return result

    # ── Node Access ──────────────────────────────────────────────

    @router.get("/nodes/{node_id}")
    async def get_node(node_id: str):
        node = server.store.get_parent_node(node_id)
        if node is None:
            # Try child node
            child = server.store.get_child_node(node_id)
            if child is None:
                raise HTTPException(status_code=404, detail="Node not found")
            return {
                "node_id": child.node_id,
                "type": "child",
                "parent_node_id": child.parent_node_id,
                "hierarchy_level": int(child.hierarchy_level),
                "summary": child.summary,
                "reasoning": child.reasoning,
                "action": child.action,
                "result": child.result,
                "step_index": child.step_index,
                "confidence": child.confidence,
            }
        return {
            "node_id": node.node_id,
            "type": "parent",
            "hierarchy_level": int(node.hierarchy_level),
            "goal": node.goal,
            "summary": node.summary,
            "confidence": node.confidence,
            "outcome": node.outcome.value,
            "success_rate": node.success_rate,
            "version": node.version,
            "child_count": len(node.child_node_ids),
            "trigger_goals": node.trigger_goals,
            "negative_constraints": node.negative_constraints,
        }

    @router.get("/nodes/{node_id}/children")
    async def get_children(node_id: str):
        children = server.store.get_child_nodes_for_parent(node_id)
        return [
            {
                "node_id": c.node_id,
                "step_index": c.step_index,
                "summary": c.summary,
                "reasoning": c.reasoning,
                "action": c.action,
                "result": c.result,
                "confidence": c.confidence,
                "is_critical_path": c.is_critical_path,
            }
            for c in children
        ]

    @router.get("/nodes/{node_id}/traverse")
    async def traverse_node(node_id: str):
        """Walk the graph from a node — return edges and connected nodes."""
        edges_out = server.store.get_edges_from(node_id)
        edges_in = server.store.get_edges_to(node_id)
        return {
            "node_id": node_id,
            "edges_out": [
                {
                    "edge_id": e.edge_id,
                    "target": e.target_node_id,
                    "type": e.edge_type.value,
                    "weight": e.weight,
                }
                for e in edges_out
            ],
            "edges_in": [
                {
                    "edge_id": e.edge_id,
                    "source": e.source_node_id,
                    "type": e.edge_type.value,
                    "weight": e.weight,
                }
                for e in edges_in
            ],
        }

    # ── Router ───────────────────────────────────────────────────

    @router.post("/route")
    async def route(req: RouteRequest):
        decision = server.router.query(req.query)
        result = {
            "path": decision.path.value,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
        }
        if decision.entry_point:
            ep = decision.entry_point
            result["entry_point"] = {
                "node_id": ep.parent_node.node_id,
                "goal": ep.parent_node.goal,
                "summary": ep.parent_node.summary,
            }
        return result

    # ── Domain Management ────────────────────────────────────────

    @router.get("/domains")
    async def list_domains():
        domains = server.store.get_domains()
        return {"domains": domains}

    @router.post("/domains/{name}/dream")
    async def domain_dream(name: str):
        engine = server.get_engine(name)
        journal = await engine.dream()
        return {
            "domain": name,
            "journal_id": journal.journal_id,
            "traces_processed": journal.traces_processed,
            "nodes_created": journal.nodes_created,
            "nodes_merged": journal.nodes_merged,
        }

    # ── ISA Version ──────────────────────────────────────────────

    @router.get("/isa/version")
    async def isa_version():
        registry = get_registry()
        return {
            "current": ISA_VERSION,
            "supported": registry.all_versions(),
        }

    # ── Stats ────────────────────────────────────────────────────

    @router.get("/stats")
    async def stats():
        data = server.store.get_stats()
        data["isa_version"] = ISA_VERSION
        return data

    # ── WebSocket: Dream Progress ────────────────────────────────

    @router.websocket("/ws/dream")
    async def ws_dream(websocket: WebSocket):
        await websocket.accept()
        try:
            data = await websocket.receive_text()
            msg = json.loads(data)
            domain = msg.get("domain", "default")

            await websocket.send_json({"event": "started", "domain": domain})
            engine = server.get_engine(domain)
            journal = await engine.dream()
            await websocket.send_json({
                "event": "completed",
                "journal_id": journal.journal_id,
                "traces_processed": journal.traces_processed,
                "nodes_created": journal.nodes_created,
                "nodes_merged": journal.nodes_merged,
                "edges_created": journal.edges_created,
                "phase_log": journal.phase_log,
            })
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as exc:
            try:
                await websocket.send_json({"event": "error", "detail": str(exc)})
            except Exception:
                pass

    return router
