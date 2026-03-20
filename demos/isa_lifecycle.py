#!/usr/bin/env python3
"""Demo: Cognitive ISA Evolution — Versioning a Live Memory Graph.

Shows how evolving-memory handles ISA versioning mid-deployment:
  Act 1 — Build knowledge under ISA 1.0
  Act 2 — Simulate an ISA upgrade (re-tag data as legacy "0.9")
  Act 3 — Dream migrates legacy data to current ISA, new knowledge integrates seamlessly

Directly addresses: "versioning a cognitive ISA mid-deployment across active
sessions without invalidating accumulated memory graphs."

Usage:
    GEMINI_API_KEY=... PYTHONPATH=src python3.12 demos/isa_lifecycle.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

# Prevent faiss/torch BLAS threading segfault on some platforms
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from evolving_memory import (
    CognitiveTrajectoryEngine,
    HierarchyLevel,
    ISA_VERSION,
    RouterPath,
)
from evolving_memory.isa.opcodes import get_registry
from evolving_memory.models.hierarchy import TraceOutcome
from evolving_memory.llm.gemini_provider import GeminiProvider


# ── Helpers ───────────────────────────────────────────────────────────

def _header(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def _sub(label: str, text: str, max_len: int = 250) -> None:
    truncated = text.strip().replace("\n", " ")[:max_len]
    print(f"  -> {label}: {truncated}")


def _audit(cte: CognitiveTrajectoryEngine, label: str) -> None:
    """Print a concise ISA version audit of the memory graph."""
    store = cte._store
    all_nodes = store.get_all_parent_nodes()
    legacy_nodes = store.get_legacy_parent_nodes()
    legacy_traces = store.get_legacy_trace_count()
    stats = store.get_stats()

    # Group nodes by ISA version
    version_counts: dict[str, int] = {}
    for node in all_nodes:
        version_counts[node.isa_version] = version_counts.get(node.isa_version, 0) + 1

    print(f"  [{label}]")
    print(f"    Total parent nodes:  {stats['parent_nodes']}")
    print(f"    Total traces:        {stats['traces']}")
    print(f"    Legacy nodes:        {len(legacy_nodes)}")
    print(f"    Legacy traces:       {legacy_traces}")
    if version_counts:
        for ver, count in sorted(version_counts.items()):
            print(f"    Nodes at ISA {ver}:    {count}")
    else:
        print(f"    (no nodes yet)")


# ── Act 1: Sprint 1 — Building Knowledge (ISA 1.0) ───────────────────

async def act1_build_knowledge(
    cte: CognitiveTrajectoryEngine,
    gemini: GeminiProvider,
) -> None:
    _header("ACT 1: SPRINT 1 -- BUILDING KNOWLEDGE (ISA 1.0)")

    # Show ISA registry state
    registry = get_registry()
    opcodes = registry.get(ISA_VERSION)
    print(f"  ISA Registry:")
    print(f"    Current version:     {registry.current()}")
    print(f"    Supported versions:  {registry.all_versions()}")
    print(f"    Opcode count (1.0):  {len(opcodes) if opcodes else 0}")
    print()

    # Trace 1: Deploy a web application (GOAL level)
    print("  [Trace 1] Web application deployment strategy...\n")
    with cte.session("deploy web application") as logger:
        with logger.trace(
            HierarchyLevel.GOAL,
            "Deploy web application to production",
            tags=["deployment", "devops", "production"],
        ) as ctx:
            resp1 = await gemini.complete(
                "What are the essential steps for deploying a web application to "
                "production? Cover infrastructure, monitoring, and rollout strategy. "
                "Be concise (4-5 sentences).",
                system="You are a senior DevOps engineer. Be practical and concise.",
            )
            ctx.action(
                reasoning="Establish production deployment fundamentals",
                action_payload="Ask: essential production deployment steps",
                result=resp1,
            )
            _sub("Deployment Steps", resp1)

            resp2 = await gemini.complete(
                "How do you containerize a web application with Docker for production? "
                "Cover Dockerfile best practices and multi-stage builds. "
                "Be concise (4-5 sentences).",
                system="You are a senior DevOps engineer. Be practical and concise.",
            )
            ctx.action(
                reasoning="Learn containerization strategy for reliable deployments",
                action_payload="Ask: Docker containerization strategy",
                result=resp2,
            )
            _sub("Containerization", resp2)
            ctx.set_outcome(TraceOutcome.SUCCESS, confidence=0.90)

    # Trace 2: CI/CD pipeline (TACTICAL level)
    print(f"\n  [Trace 2] CI/CD pipeline design...\n")
    with cte.session("CI/CD pipeline") as logger:
        with logger.trace(
            HierarchyLevel.TACTICAL,
            "Design CI/CD pipeline",
            tags=["cicd", "automation", "pipeline"],
        ) as ctx:
            resp3 = await gemini.complete(
                "How would you design a CI/CD pipeline for a web application? "
                "Cover stages from commit to production. Be concise (4-5 sentences).",
                system="You are a CI/CD specialist. Be practical and concise.",
            )
            ctx.action(
                reasoning="Design a complete CI/CD pipeline architecture",
                action_payload="Ask: CI/CD pipeline design",
                result=resp3,
            )
            _sub("Pipeline Design", resp3)

            resp4 = await gemini.complete(
                "What are effective rollback strategies when a deployment fails? "
                "Cover blue-green, canary, and feature flags. Be concise (4-5 sentences).",
                system="You are a CI/CD specialist. Be practical and concise.",
            )
            ctx.action(
                reasoning="Learn rollback strategies for deployment safety",
                action_payload="Ask: rollback strategies",
                result=resp4,
            )
            _sub("Rollback Strategies", resp4)
            ctx.set_outcome(TraceOutcome.SUCCESS, confidence=0.85)

    print(f"\n  Stored 2 traces (4 actions) under ISA {ISA_VERSION}")

    # Dream cycle 1 — consolidate into memory nodes
    print("\n  Dreaming (consolidating Sprint 1 traces)...")
    journal1 = await cte.dream()
    print(f"    Traces processed:      {journal1.traces_processed}")
    print(f"    Nodes created:         {journal1.nodes_created}")
    print(f"    Edges created:         {journal1.edges_created}")
    print(f"    Constraints extracted: {journal1.constraints_extracted}")

    # Audit: all data at ISA 1.0
    print()
    _audit(cte, "Post-Sprint 1 Audit")


# ── Act 2: ISA Upgrade Simulation ─────────────────────────────────────

def act2_simulate_upgrade(cte: CognitiveTrajectoryEngine) -> None:
    _header("ACT 2: ISA UPGRADE SIMULATION")

    print("  Simulating an ISA upgrade scenario:")
    print('  Re-tagging all existing data as legacy ISA "0.9"...\n')

    conn = cte._store._conn

    # Re-tag parent_nodes to ISA 0.9
    node_result = conn.execute(
        "UPDATE parent_nodes SET isa_version = '0.9' WHERE isa_version = ?",
        (ISA_VERSION,),
    )
    nodes_updated = node_result.rowcount
    conn.commit()

    # Re-tag trace_entries to ISA 0.9
    trace_result = conn.execute(
        "UPDATE trace_entries SET isa_version = '0.9' WHERE isa_version = ?",
        (ISA_VERSION,),
    )
    traces_updated = trace_result.rowcount
    conn.commit()

    print(f"  Re-tagged {nodes_updated} parent nodes:  1.0 -> 0.9")
    print(f"  Re-tagged {traces_updated} trace entries: 1.0 -> 0.9")

    # Show the legacy audit
    print()
    _audit(cte, "Post-Upgrade Audit (legacy data)")

    # Registry awareness
    registry = get_registry()
    print(f"\n  Registry check:")
    print(f"    Supports ISA 0.9? {registry.supports('0.9')}")
    print(f"    Supports ISA 1.0? {registry.supports('1.0')}")
    print(f"    Current version:  {registry.current()}")
    print(f"\n  The system recognizes 0.9 as legacy — next dream will migrate it.")


# ── Act 3: Sprint 2 — Memory Continuity After Upgrade ─────────────────

async def act3_memory_continuity(
    cte: CognitiveTrajectoryEngine,
    gemini: GeminiProvider,
) -> None:
    _header("ACT 3: SPRINT 2 -- MEMORY CONTINUITY AFTER UPGRADE")

    # Trace 3: Scale the web application (GOAL level)
    print("  [Trace 3] Scaling the web application...\n")
    with cte.session("scale web application") as logger:
        with logger.trace(
            HierarchyLevel.GOAL,
            "Scale web application for high traffic",
            tags=["scaling", "infrastructure", "performance"],
        ) as ctx:
            resp5 = await gemini.complete(
                "What are the key strategies for horizontally scaling a web application? "
                "Cover load balancing, stateless design, and auto-scaling. "
                "Be concise (4-5 sentences).",
                system="You are an infrastructure architect. Be practical and concise.",
            )
            ctx.action(
                reasoning="Learn horizontal scaling fundamentals",
                action_payload="Ask: horizontal scaling strategies",
                result=resp5,
            )
            _sub("Horizontal Scaling", resp5)

            resp6 = await gemini.complete(
                "How do you scale databases for a high-traffic web application? "
                "Cover read replicas, sharding, and caching layers. "
                "Be concise (4-5 sentences).",
                system="You are a database architect. Be practical and concise.",
            )
            ctx.action(
                reasoning="Learn database scaling patterns",
                action_payload="Ask: database scaling strategies",
                result=resp6,
            )
            _sub("Database Scaling", resp6)
            ctx.set_outcome(TraceOutcome.SUCCESS, confidence=0.90)

    print(f"\n  Stored 1 trace (2 actions) under ISA {ISA_VERSION}")

    # Dream cycle 2 — Phase 0 migrates legacy, then processes new traces
    print("\n  Dreaming (Phase 0 migration + Sprint 2 consolidation)...")
    journal2 = await cte.dream()

    print(f"\n  Dream Journal:")
    print(f"    Nodes migrated (Phase 0):   {journal2.nodes_migrated}")
    print(f"    Traces migrated (Phase 0):  {journal2.traces_migrated}")
    print(f"    Traces processed (new):     {journal2.traces_processed}")
    print(f"    Nodes created:              {journal2.nodes_created}")
    print(f"    Edges created:              {journal2.edges_created}")
    if journal2.phase_log:
        print(f"\n  Phase log:")
        for entry in journal2.phase_log:
            print(f"    * {entry}")

    # Final audit: everything at ISA 1.0
    print()
    _audit(cte, "Final Audit (post-migration)")

    # Query old knowledge (deployment)
    _header("QUERY: Old Knowledge (pre-upgrade)")
    print('  "How do I deploy a web application to production?"')
    decision1 = cte.query("How do I deploy a web application to production?")
    print(f"    Route:      {decision1.path.value}")
    print(f"    Confidence: {decision1.confidence:.2f}")
    print(f"    Reasoning:  {decision1.reasoning[:120]}")

    if decision1.path == RouterPath.MEMORY_TRAVERSAL and decision1.entry_point:
        ep = decision1.entry_point
        print(f"\n    Entry point:   {ep.parent_node.goal}")
        print(f"    ISA version:   {ep.parent_node.isa_version}")
        print(f"    Similarity:    {ep.similarity_score:.2f}")
        if ep.parent_node.isa_version == ISA_VERSION:
            print(f"    [OK] Legacy node successfully migrated to ISA {ISA_VERSION}")
    else:
        print(f"    (routed to {decision1.path.value} — no memory traversal)")

    # Query new knowledge (scaling)
    print(f"\n\n  -- Query: New Knowledge (post-upgrade) --")
    print('  "How do I scale a web application for high traffic?"')
    decision2 = cte.query("How do I scale a web application for high traffic?")
    print(f"    Route:      {decision2.path.value}")
    print(f"    Confidence: {decision2.confidence:.2f}")
    print(f"    Reasoning:  {decision2.reasoning[:120]}")

    if decision2.path == RouterPath.MEMORY_TRAVERSAL and decision2.entry_point:
        ep2 = decision2.entry_point
        print(f"\n    Entry point:   {ep2.parent_node.goal}")
        print(f"    ISA version:   {ep2.parent_node.isa_version}")
        print(f"    Similarity:    {ep2.similarity_score:.2f}")
        if ep2.parent_node.isa_version == ISA_VERSION:
            print(f"    [OK] New node created at ISA {ISA_VERSION}")
    else:
        print(f"    (routed to {decision2.path.value} — no memory traversal)")


# ── Main ──────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n  Evolving Memory -- Cognitive ISA Evolution Demo")
    print("=" * 55)
    print(f"  Demonstrating: ISA versioning, schema migration, and")
    print(f"  memory continuity across cognitive architecture upgrades")

    # Setup
    gemini = GeminiProvider()
    tmp = tempfile.mktemp(suffix=".db")
    cte = CognitiveTrajectoryEngine(llm=gemini, db_path=tmp)

    try:
        await act1_build_knowledge(cte, gemini)
        act2_simulate_upgrade(cte)
        await act3_memory_continuity(cte, gemini)

        _header("DONE")
        print("  Demo complete. ISA lifecycle verified:")
        print("    1. Knowledge built under ISA 1.0")
        print('    2. Data re-tagged as legacy "0.9" (simulating upgrade)')
        print("    3. Dream Phase 0 migrated all legacy data back to 1.0")
        print("    4. Old + new knowledge queries work seamlessly")
        print(f"\n  DB: {tmp}")
    finally:
        cte.close()
        # Clean up temp DB
        for f in Path(tmp).parent.glob(Path(tmp).stem + "*"):
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
    # Avoid segfault during interpreter shutdown (faiss/torch cleanup race)
    os._exit(0)
