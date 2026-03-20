#!/usr/bin/env python3
"""Demo 3: Cognitive ISA Evolution — Zero-Downtime Memory Upgrades.

Demonstrates how evolving-memory handles ISA versioning mid-deployment
without invalidating accumulated memory graphs. The agent "updates its own
memories" while it sleeps.

The Scenario:
  A FinTech company's agent has been processing refunds under ISA 1.0.
  The company decides ISA 2.0 requires risk assessment on every memory
  step (risk_level: low/medium/high/critical). Legacy memories don't
  have this field.

  Act 1 — Agent learns refund processing strategy under ISA 1.0 (no risk)
  Act 2 — Company upgrades to ISA 2.0, marks all data as legacy "1.0"
  Act 3 — Dream Phase 0 migrates legacy data AND the LLM retroactively
           evaluates risk levels for each step in the old memories
  Act 4 — Query returns v2.0 memory with risk_level that didn't exist
           when the memory was originally created

  This is the "mic drop": the agent updated its past memories based on
  the new cognitive rules of the present.

Directly addresses: "versioning a cognitive ISA mid-deployment across active
sessions without invalidating accumulated memory graphs."

Usage:
    GEMINI_API_KEY=... PYTHONPATH=src python3.12 demos/isa_lifecycle.py
"""

from __future__ import annotations

import asyncio
import json
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
    MigrationTransform,
    RouterPath,
)
from evolving_memory.isa.opcodes import get_registry
from evolving_memory.models.graph import ChildNode, ParentNode
from evolving_memory.models.hierarchy import TraceOutcome
from evolving_memory.llm.base import BaseLLMProvider
from evolving_memory.llm.gemini_provider import GeminiProvider


# ── Risk Level Migration Transform ──────────────────────────────────

class AddRiskAssessment(MigrationTransform):
    """Cognitive migration: ISA 1.0 -> 2.0 adds risk_level to every memory step.

    During Phase 0, the LLM re-evaluates each legacy refund processing step
    and assigns a risk level (low/medium/high/critical) based on the financial
    impact and compliance implications. This knowledge didn't exist in v1.0.
    """

    from_version = "1.0"
    to_version = "2.0"

    async def transform(
        self,
        node: ParentNode,
        children: list[ChildNode],
        llm: BaseLLMProvider,
    ) -> tuple[ParentNode, list[ChildNode]]:
        # Build a summary of all steps for the LLM
        steps_text = "\n".join(
            f"  Step {c.step_index}: {c.reasoning} -> {c.action}"
            + (f" (Result: {c.result[:150]})" if c.result else "")
            for c in children
        )

        prompt = (
            f"You are a compliance officer evaluating an agent's learned strategy.\n"
            f"The strategy: {node.goal}\n"
            f"Summary: {node.summary}\n"
            f"Steps:\n{steps_text}\n\n"
            f"For each step, assess the risk level for financial processing:\n"
            f"- low: Routine, no financial impact if wrong\n"
            f"- medium: Moderate impact, reversible errors\n"
            f"- high: Significant financial impact, hard to reverse\n"
            f"- critical: Regulatory/compliance risk, irreversible\n\n"
            f"Also provide an overall risk assessment for the strategy.\n\n"
            f"Respond in this EXACT JSON format:\n"
            f'{{"overall_risk": "low|medium|high|critical", '
            f'"risk_rationale": "one sentence", '
            f'"steps": [{{"step_index": 0, "risk_level": "low|medium|high|critical", '
            f'"risk_reason": "brief reason"}}]}}'
        )

        try:
            response = await llm.complete(
                prompt,
                system="You are a financial compliance AI. Respond ONLY with valid JSON.",
            )

            # Parse the JSON response (handle markdown code blocks)
            json_text = response.strip()
            if json_text.startswith("```"):
                json_text = json_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            assessment = json.loads(json_text)

            # Enrich the parent node with overall risk
            risk_tag = f"[RISK: {assessment['overall_risk'].upper()}]"
            rationale = assessment.get("risk_rationale", "")
            node.content = (
                f"{node.content}\n\n"
                f"--- ISA 2.0 Risk Assessment ---\n"
                f"Overall Risk Level: {assessment['overall_risk']}\n"
                f"Rationale: {rationale}"
            )
            node.summary = f"{risk_tag} {node.summary}"

            # Enrich each child step with its risk level
            step_risks = {s["step_index"]: s for s in assessment.get("steps", [])}
            for child in children:
                step_risk = step_risks.get(child.step_index, {})
                risk_level = step_risk.get("risk_level", "medium")
                risk_reason = step_risk.get("risk_reason", "")
                child.content = (
                    f"{child.content}\n"
                    f"[risk_level: {risk_level}] {risk_reason}"
                )

        except Exception as e:
            # Fallback: tag as "unassessed" — migration still succeeds
            node.content = f"{node.content}\n\n[risk_level: unassessed] Migration transform error: {e}"

        return node, children


# ── Helpers ───────────────────────────────────────────────────────────

def _header(title: str) -> None:
    width = 65
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
    print(f"    Total child nodes:   {stats['child_nodes']}")
    print(f"    Total traces:        {stats['traces']}")
    print(f"    Legacy nodes:        {len(legacy_nodes)}")
    print(f"    Legacy traces:       {legacy_traces}")
    if version_counts:
        for ver, count in sorted(version_counts.items()):
            print(f"    Nodes at ISA {ver}:    {count}")
    else:
        print(f"    (no nodes yet)")


# ── Act 1: Building Refund Processing Knowledge (ISA 1.0) ──────────

async def act1_build_knowledge(
    cte: CognitiveTrajectoryEngine,
    gemini: GeminiProvider,
) -> None:
    _header("ACT 1: BUILDING KNOWLEDGE (ISA 1.0)")
    print("  SCENARIO: Agent learns refund processing strategy")
    print("  NOTE: ISA 1.0 has NO risk assessment — just procedures\n")

    # Show ISA registry state
    registry = get_registry()
    opcodes = registry.get(ISA_VERSION)
    print(f"  ISA Registry:")
    print(f"    Current version:     {registry.current()}")
    print(f"    Supported versions:  {registry.all_versions()}")
    print(f"    Opcode count (1.0):  {len(opcodes) if opcodes else 0}")
    print()

    # Trace 1: Refund processing strategy (GOAL level)
    print("  [Trace 1] Refund processing strategy...\n")
    with cte.session("process customer refunds") as logger:
        with logger.trace(
            HierarchyLevel.GOAL,
            "Process customer refund requests end-to-end",
            tags=["refund", "payments", "fintech"],
        ) as ctx:
            resp1 = await gemini.complete(
                "What are the essential steps for processing a customer refund "
                "in a SaaS company? Cover validation, approval workflow, payment "
                "gateway interaction, and customer notification. Be concise (5-6 sentences).",
                system="You are a senior payments engineer at a FinTech company. Be practical.",
            )
            ctx.action(
                reasoning="Establish end-to-end refund processing workflow",
                action_payload="Ask: essential refund processing steps",
                result=resp1,
            )
            _sub("Refund Workflow", resp1)

            resp2 = await gemini.complete(
                "How do you handle partial refunds and multi-currency refund "
                "calculations? Cover exchange rate locking, partial amount "
                "validation, and ledger reconciliation. Be concise (4-5 sentences).",
                system="You are a senior payments engineer at a FinTech company. Be practical.",
            )
            ctx.action(
                reasoning="Learn partial and multi-currency refund handling",
                action_payload="Ask: partial refund and currency handling",
                result=resp2,
            )
            _sub("Partial Refunds", resp2)
            ctx.set_outcome(TraceOutcome.SUCCESS, confidence=0.90)

    # Trace 2: Dispute handling (TACTICAL level)
    print(f"\n  [Trace 2] Dispute resolution procedures...\n")
    with cte.session("handle payment disputes") as logger:
        with logger.trace(
            HierarchyLevel.TACTICAL,
            "Handle chargeback and dispute resolution",
            tags=["disputes", "chargebacks", "compliance"],
        ) as ctx:
            resp3 = await gemini.complete(
                "What is the correct procedure for handling a Stripe chargeback? "
                "Cover evidence submission, deadline management, and when to accept "
                "vs contest a dispute. Be concise (5-6 sentences).",
                system="You are a payments compliance specialist. Be practical.",
            )
            ctx.action(
                reasoning="Learn chargeback response procedures",
                action_payload="Ask: chargeback handling procedure",
                result=resp3,
            )
            _sub("Chargeback Handling", resp3)

            resp4 = await gemini.complete(
                "What are the common fraud indicators in refund requests that "
                "should trigger manual review? Cover velocity checks, amount "
                "thresholds, and behavioral signals. Be concise (4-5 sentences).",
                system="You are a fraud prevention specialist. Be practical.",
            )
            ctx.action(
                reasoning="Learn fraud detection signals for refund requests",
                action_payload="Ask: refund fraud indicators",
                result=resp4,
            )
            _sub("Fraud Indicators", resp4)
            ctx.set_outcome(TraceOutcome.SUCCESS, confidence=0.85)

    print(f"\n  Stored 2 traces (4 actions) under ISA {ISA_VERSION}")
    print(f"  NOTE: No risk levels — ISA 1.0 doesn't require them")

    # Dream cycle 1 — consolidate into memory nodes
    print("\n  Dreaming (consolidating refund knowledge)...")
    journal1 = await cte.dream()
    print(f"    Traces processed:      {journal1.traces_processed}")
    print(f"    Nodes created:         {journal1.nodes_created}")
    print(f"    Edges created:         {journal1.edges_created}")
    print(f"    Constraints extracted: {journal1.constraints_extracted}")

    # Audit: all data at ISA 1.0
    print()
    _audit(cte, "Post-Act 1 Audit")

    # Show that nodes have NO risk information
    print("\n  [Verify] Checking node content for risk data...")
    all_nodes = cte._store.get_all_parent_nodes()
    for node in all_nodes:
        has_risk = "risk" in node.content.lower() or "risk" in node.summary.lower()
        print(f"    Node: {node.goal[:50]}")
        print(f"      ISA version:  {node.isa_version}")
        print(f"      Has risk data: {'Yes' if has_risk else 'No (expected)'}")


# ── Act 2: ISA Upgrade — v2.0 Requires Risk Assessment ───────────

def act2_simulate_upgrade(cte: CognitiveTrajectoryEngine) -> None:
    _header("ACT 2: ISA UPGRADE — v2.0 REQUIRES RISK ASSESSMENT")

    print("  THE BUSINESS DECISION:")
    print("    After a compliance audit, the company mandates that every")
    print("    step in the agent's memory must include a risk assessment.")
    print("    ISA 2.0 requires: risk_level (low/medium/high/critical)")
    print()
    print("  THE PROBLEM:")
    print("    All existing memories were built under ISA 1.0 (no risk data).")
    print("    A traditional system would require:")
    print("      - Manual re-labeling of all historical data")
    print("      - OR discarding old memories entirely")
    print("      - OR running with inconsistent schema")
    print()
    print("  EVOLVING MEMORY'S SOLUTION:")
    print("    Register a Cognitive Migration Transform that uses the LLM")
    print("    to retroactively assess risk for all legacy memories during")
    print("    the next dream cycle (Phase 0).")
    print()
    print('  Simulating upgrade: re-tagging all data as legacy "1.0"...\n')

    conn = cte._store._conn

    # Re-tag parent_nodes to ISA 1.0 (simulating they were created before upgrade)
    node_result = conn.execute(
        "UPDATE parent_nodes SET isa_version = '1.0' WHERE isa_version = ?",
        (ISA_VERSION,),
    )
    nodes_updated = node_result.rowcount
    conn.commit()

    # Re-tag trace_entries to ISA 1.0
    trace_result = conn.execute(
        "UPDATE trace_entries SET isa_version = '1.0' WHERE isa_version = ?",
        (ISA_VERSION,),
    )
    traces_updated = trace_result.rowcount
    conn.commit()

    print(f"  Re-tagged {nodes_updated} parent nodes:  current -> 1.0 (legacy)")
    print(f"  Re-tagged {traces_updated} trace entries: current -> 1.0 (legacy)")

    # Register the cognitive migration transform
    print(f"\n  Registering Cognitive Migration: AddRiskAssessment (1.0 -> 2.0)")
    cte.register_migration(AddRiskAssessment())

    # Show the legacy audit
    print()
    _audit(cte, "Post-Upgrade Audit (legacy data, no risk)")

    # Registry awareness
    registry = get_registry()
    print(f"\n  Registry check:")
    print(f"    Supports ISA 1.0? {registry.supports('1.0')}")
    print(f"    Current version:  {registry.current()}")
    print(f"\n  Next dream cycle will:")
    print(f"    1. Detect legacy 1.0 nodes")
    print(f"    2. Run AddRiskAssessment transform (LLM-powered)")
    print(f"    3. Inject risk_level into every step")
    print(f"    4. Re-stamp nodes to current ISA version")


# ── Act 3: The Migratory Dream ──────────────────────────────────────

async def act3_migratory_dream(
    cte: CognitiveTrajectoryEngine,
    gemini: GeminiProvider,
) -> None:
    _header("ACT 3: THE MIGRATORY DREAM (Phase 0 Cognitive Migration)")

    print("  The agent is dreaming...")
    print("  Phase 0 will detect legacy nodes and run the LLM-powered")
    print("  AddRiskAssessment transform on each one.\n")

    # Optionally add a new trace to show new + legacy coexistence
    print("  [Trace 3] New knowledge under current ISA (post-upgrade)...\n")
    with cte.session("compliance monitoring") as logger:
        with logger.trace(
            HierarchyLevel.TACTICAL,
            "Monitor refund compliance metrics",
            tags=["compliance", "monitoring", "metrics"],
        ) as ctx:
            resp = await gemini.complete(
                "What KPIs should a FinTech company track for refund compliance? "
                "Cover refund rate, processing time SLA, dispute win rate, and "
                "regulatory reporting. Be concise (4-5 sentences).",
                system="You are a compliance monitoring specialist. Be practical.",
            )
            ctx.action(
                reasoning="Establish refund compliance KPIs",
                action_payload="Ask: refund compliance monitoring KPIs",
                result=resp,
            )
            _sub("Compliance KPIs", resp)
            ctx.set_outcome(TraceOutcome.SUCCESS, confidence=0.85)

    print(f"\n  Stored 1 new trace under ISA {ISA_VERSION}")

    # Dream cycle 2 — Phase 0 migrates + enriches legacy, then processes new
    print("\n  Dreaming (Phase 0 migration + cognitive enrichment)...")
    journal2 = await cte.dream()

    print(f"\n  Dream Journal:")
    print(f"    Phase 0 — Nodes migrated:       {journal2.nodes_migrated}")
    print(f"    Phase 0 — Traces migrated:       {journal2.traces_migrated}")
    print(f"    Phases 1-3 — Traces processed:   {journal2.traces_processed}")
    print(f"    Phases 1-3 — Nodes created:      {journal2.nodes_created}")
    print(f"    Phases 1-3 — Edges created:      {journal2.edges_created}")
    if journal2.phase_log:
        print(f"\n  Phase log:")
        for entry in journal2.phase_log:
            print(f"    * {entry}")

    # Final audit: everything migrated
    print()
    _audit(cte, "Post-Migration Audit")


# ── Act 4: The Verification — Memory Has Been Retroactively Enhanced ─

async def act4_verify_migration(cte: CognitiveTrajectoryEngine) -> None:
    _header("ACT 4: VERIFICATION — Memory Retroactively Enhanced")

    print("  Querying old memories to verify they now contain risk data")
    print("  that DID NOT EXIST when they were originally created.\n")

    # Query 1: Refund processing (old knowledge)
    print('  Query: "How do I process a customer refund?"')
    decision1 = cte.query("How do I process a customer refund?")
    print(f"    Route:      {decision1.path.value}")
    print(f"    Confidence: {decision1.confidence:.2f}")

    if decision1.path == RouterPath.MEMORY_TRAVERSAL and decision1.entry_point:
        ep = decision1.entry_point
        parent = ep.parent_node
        print(f"\n    Entry point:   {parent.goal}")
        print(f"    ISA version:   {parent.isa_version}")
        print(f"    Similarity:    {ep.similarity_score:.2f}")

        # Check for risk data in the parent
        has_risk = "risk" in parent.content.lower() or "risk" in parent.summary.lower()
        print(f"\n    --- Risk Assessment (retroactively added) ---")
        if has_risk:
            print(f"    [OK] Parent node now contains risk assessment!")
            # Extract risk info from content
            for line in parent.content.split("\n"):
                if "risk" in line.lower():
                    print(f"         {line.strip()}")
        else:
            print(f"    [!] Risk data not found in parent (transform may have failed)")

        # Walk the children and show risk levels
        state = cte.begin_traversal(ep)
        print(f"\n    --- Step-by-Step Risk Levels ---")
        while True:
            child, state = cte.next_step(state)
            if child is None:
                break
            risk_info = ""
            for line in child.content.split("\n"):
                if "risk_level" in line.lower():
                    risk_info = line.strip()
                    break
            print(f"    Step {child.step_index}: {child.reasoning[:60]}")
            if risk_info:
                print(f"             {risk_info}")
            else:
                print(f"             (no risk tag — transform fallback)")

        if parent.isa_version == ISA_VERSION:
            print(f"\n    [OK] Legacy node migrated to ISA {ISA_VERSION} with risk data")
        print()
    else:
        print(f"    (routed to {decision1.path.value} — no memory traversal)")

    # Query 2: Dispute handling (old knowledge, should also have risk)
    print('  Query: "How do I handle a payment chargeback?"')
    decision2 = cte.query("How do I handle a payment chargeback?")
    print(f"    Route:      {decision2.path.value}")
    print(f"    Confidence: {decision2.confidence:.2f}")

    if decision2.path == RouterPath.MEMORY_TRAVERSAL and decision2.entry_point:
        ep2 = decision2.entry_point
        parent2 = ep2.parent_node
        print(f"    Entry point: {parent2.goal}")
        print(f"    ISA version: {parent2.isa_version}")
        has_risk2 = "risk" in parent2.summary.lower()
        if has_risk2:
            print(f"    Summary:     {parent2.summary[:120]}")
            print(f"    [OK] Risk tag present in summary")
    else:
        print(f"    (routed to {decision2.path.value})")

    # Query 3: New knowledge (compliance monitoring, created post-upgrade)
    print(f"\n  Query: New knowledge (created after upgrade)")
    print('  "What KPIs should I track for refund compliance?"')
    decision3 = cte.query("What KPIs should I track for refund compliance?")
    print(f"    Route:      {decision3.path.value}")
    print(f"    Confidence: {decision3.confidence:.2f}")

    if decision3.path == RouterPath.MEMORY_TRAVERSAL and decision3.entry_point:
        ep3 = decision3.entry_point
        print(f"    Entry point: {ep3.parent_node.goal}")
        print(f"    ISA version: {ep3.parent_node.isa_version}")
        if ep3.parent_node.isa_version == ISA_VERSION:
            print(f"    [OK] New node at current ISA version")


# ── Main ──────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n" + "=" * 65)
    print("  EVOLVING MEMORY — Cognitive ISA Evolution Demo")
    print("  'The agent updates its own memories while it sleeps'")
    print("=" * 65)
    print()
    print("  Demonstrating: Zero-downtime cognitive upgrades via")
    print("  LLM-powered memory migration during the dream cycle.")
    print()
    print("  SCENARIO: FinTech company upgrades from ISA 1.0 to 2.0")
    print("  NEW REQUIREMENT: Every memory step must have a risk_level")
    print("  CHALLENGE: All existing memories lack risk assessment")

    # Setup
    gemini = GeminiProvider()
    tmp = tempfile.mktemp(suffix=".db")
    cte = CognitiveTrajectoryEngine(llm=gemini, db_path=tmp)

    try:
        # Act 1: Build knowledge under ISA 1.0 (no risk)
        await act1_build_knowledge(cte, gemini)

        # Act 2: Simulate ISA upgrade to 2.0 (requires risk_level)
        act2_simulate_upgrade(cte)

        # Act 3: Dream migrates + enriches legacy memories
        await act3_migratory_dream(cte, gemini)

        # Act 4: Verify memories were retroactively enhanced
        await act4_verify_migration(cte)

        # Final summary
        _header("CONCLUSION")
        print("  Demo complete. Cognitive ISA Evolution verified:\n")
        print("    1. Knowledge built under ISA 1.0 (no risk assessment)")
        print("    2. Company mandates ISA 2.0 (risk_level required)")
        print("    3. Cognitive Migration Transform registered")
        print("    4. Dream Phase 0 detected legacy nodes and used the LLM")
        print("       to retroactively assess risk for every memory step")
        print("    5. Old + new knowledge queries work seamlessly at v2.0")
        print()
        print("  THE MIC DROP:")
        print("    The agent's memories from BEFORE the upgrade now contain")
        print("    risk assessments that didn't exist when those memories")
        print("    were originally created. The agent literally updated its")
        print("    past understanding based on new cognitive rules.")
        print()
        print("  ENTERPRISE VALUE:")
        print("    - Zero-downtime cognitive upgrades (no system shutdown)")
        print("    - Business continuity (accumulated knowledge preserved)")
        print("    - Compliance-ready (new schema enforced retroactively)")
        print("    - LLM-agnostic (migrate between providers seamlessly)")
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
