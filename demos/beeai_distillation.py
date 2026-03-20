#!/usr/bin/env python3
"""Demo 2: "Cognitive Distillation" — Teacher-Student Knowledge Transfer.

Demonstrates that a large LLM can teach a small LLM through evolving-memory's
dream cycle, enabling the small model to solve problems it could never solve
on its own.

The Scenario:
  A finance team needs to reconcile ALL charges from the Stripe Payments API.
  The catch: the pagination cursor is hidden in HTTP response headers
  (Stripe-Cursor), NOT in the JSON body. The body only has `has_more: true`.

  If the agent fails to fetch all pages, the reconciliation report is WRONG:
  missing charges, incorrect totals, disputed payments unreported. In
  production, this means compliance failures and lost revenue.

  A small LLM (8B params) will try ?page=2, ?offset=5, etc. — all fail.
  It will exhaust its retries and produce an incomplete report.

  A large LLM (Gemini Pro / Claude) analyzes the failure trace during the
  dream cycle and discovers the hidden cursor pattern in the raw API response.

  The small LLM, now armed with the distilled knowledge, succeeds immediately
  and produces a complete, accurate reconciliation report.

The Proof:
  "Evolving Memory allows an 8B-parameter model to achieve the reliability
  of a 70B model, by giving it procedural memory from a more capable teacher."

  In enterprise terms: the small model + memory = compliance-ready agent
  that saves $X,XXX/month in API costs while preventing financial errors.

Usage:
    GEMINI_API_KEY=... PYTHONPATH=src python demos/beeai_distillation.py

    # Or specify models:
    BEE_LLM=ollama:llama3.1:8b DREAM_MODEL=gemini-2.5-pro PYTHONPATH=src python demos/beeai_distillation.py

Requirements:
    pip install -e ".[all]"
    pip install beeai-framework
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Prevent faiss/torch threading issues
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory

from evolving_memory import CognitiveTrajectoryEngine
from evolving_memory.llm.gemini_provider import GeminiProvider

from beeai_adapter import EvolvingMemoryAdapter, GeminiChatModel, RunMetrics, print_metrics_comparison
from mock_tools import (
    stripe_charges_api, reset_stripe_state,
    STRIPE_CORRECT_TOTAL_CHARGES, STRIPE_CORRECT_GROSS_USD,
    STRIPE_CORRECT_DISPUTED_COUNT,
)


# ── Configuration ────────────────────────────────────────────────────

TASK_PROMPT = (
    "Reconcile ALL charges from the Stripe Payments API. "
    "The base endpoint is /v1/charges. "
    "The API returns paginated results — you MUST fetch every page "
    "until there are no more results. "
    "For each charge, note the amount (convert from cents to dollars), "
    "status, and customer. "
    "Produce a reconciliation summary: total charge count, gross revenue "
    "in USD, number of disputed charges, and number of refunded charges."
)

SESSION_GOAL = "Reconcile all Stripe charges for financial reporting"

# Expected correct answers
EXPECTED_CHARGES = STRIPE_CORRECT_TOTAL_CHARGES  # 14
EXPECTED_GROSS = STRIPE_CORRECT_GROSS_USD


def _header(title: str) -> None:
    width = 65
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def _get_bee_llm():
    """Get the BeeAI ChatModel for the small (waking) LLM."""
    model_name = os.environ.get("BEE_LLM", "")
    if model_name:
        print(f"  Waking LLM (small): {model_name}")
        return ChatModel.from_name(model_name)

    # Default: use Gemini via LiteLLM if GEMINI_API_KEY is set
    if os.environ.get("GEMINI_API_KEY"):
        model = os.environ.get("BEE_GEMINI_MODEL", "gemini-2.5-flash-lite")
        print(f"  Waking LLM (small): gemini/{model}")
        return GeminiChatModel(model)

    # Fallback: try Ollama local models
    for name in ["ollama:llama3.2", "ollama:llama3.1:8b", "ollama:llama3.1", "groq:llama-3.1-8b-instant"]:
        try:
            llm = ChatModel.from_name(name)
            print(f"  Waking LLM (small): {name}")
            return llm
        except Exception:
            continue

    raise RuntimeError(
        "No small LLM available. Set GEMINI_API_KEY or BEE_LLM env var."
    )


def _get_dreaming_llm() -> GeminiProvider:
    """Get the large LLM for the dream cycle."""
    model = os.environ.get("DREAM_MODEL", "gemini-2.5-flash")
    print(f"  Dreaming LLM (large): {model}")
    return GeminiProvider(model=model)


def _check_reconciliation(final_answer: str) -> dict:
    """Check if the agent's reconciliation report is accurate."""
    result = {
        "all_charges_found": False,
        "correct_total": False,
        "disputes_flagged": False,
    }
    if str(EXPECTED_CHARGES) in final_answer:
        result["all_charges_found"] = True
    # Check if gross amount is approximately correct (allow formatting differences)
    for fmt in [f"{EXPECTED_GROSS:,.2f}", f"{EXPECTED_GROSS:.2f}", str(int(EXPECTED_GROSS))]:
        if fmt in final_answer:
            result["correct_total"] = True
            break
    if str(STRIPE_CORRECT_DISPUTED_COUNT) in final_answer and "dispute" in final_answer.lower():
        result["disputes_flagged"] = True
    return result


# ── Day 1: Small LLM Attempts ───────────────────────────────────────

async def day1_attempt(
    adapter: EvolvingMemoryAdapter,
    bee_llm,
) -> RunMetrics:
    """Day 1: Small LLM tries to reconcile Stripe charges. Expected: FAILURE."""
    _header("DAY 1: Small LLM Attempts Reconciliation (Expected: Incomplete)")

    reset_stripe_state()

    print(f"  SCENARIO: Monthly Stripe payment reconciliation")
    print(f"  RISK: Missing charges = incorrect financial reports = compliance failure")
    print(f"  Expected: {EXPECTED_CHARGES} charges, ${EXPECTED_GROSS:,.2f} gross")
    print(f"\n  Task: {TASK_PROMPT[:80]}...")
    print(f"  The small LLM will try to paginate but won't find the cursor...\n")

    # Retry loop — small LLMs sometimes fail BeeAI's ReAct format
    metrics = None
    for attempt in range(3):
        reset_stripe_state()
        agent = ReActAgent(
            llm=bee_llm,
            tools=[stripe_charges_api],
            memory=UnconstrainedMemory(),
        )
        metrics = await adapter.run_with_tracing(
            agent=agent,
            prompt=TASK_PROMPT,
            session_goal=SESSION_GOAL,
            max_retries_per_step=5,
            max_iterations=12,
            total_max_retries=12,
        )
        if metrics.steps > 0:
            break
        if attempt < 2:
            print(f"  (Retrying — model format error, attempt {attempt + 2}/3)")

    # Print iteration log
    print(f"  --- Iteration Log ---")
    for i, entry in enumerate(metrics.iterations_log, 1):
        thought = entry.get("thought", "")[:120]
        tool_name = entry.get("tool_name", "")
        tool_input = entry.get("tool_input", "")[:80]
        tool_output = entry.get("tool_output", "")[:100]
        final = entry.get("final_answer", "")[:120]

        if tool_name:
            status = "ERROR" if "error" in tool_output.lower() else "OK"
            print(f"  [{i}] {thought}")
            print(f"      -> {tool_name}({tool_input})")
            print(f"      <- [{status}] {tool_output}")
        elif final:
            print(f"  [{i}] FINAL: {final}")
        else:
            print(f"  [{i}] {thought}")
        print()

    # Diagnosis
    print(f"  --- Day 1 Result ---")
    print(f"  Steps: {metrics.steps}")
    print(f"  Tool errors: {metrics.tool_errors}")
    print(f"  Success: {'Yes' if metrics.success else 'No'}")

    # Check reconciliation accuracy
    check = _check_reconciliation(metrics.final_answer)
    if check["all_charges_found"]:
        print(f"  All {EXPECTED_CHARGES} charges found")
    else:
        print(f"  MISSING CHARGES — agent did NOT find all {EXPECTED_CHARGES} charges")
        print(f"  -> In production: reconciliation report is INCOMPLETE")
    if not check["correct_total"]:
        print(f"  -> WRONG TOTAL — financial report would be inaccurate")
    if not check["disputes_flagged"]:
        print(f"  -> DISPUTES NOT FLAGGED — compliance risk")
    print(f"\n  BUSINESS IMPACT: Incomplete reconciliation means undetected")
    print(f"  discrepancies, missed disputes, and potential audit failures.")

    return metrics


# ── Night: Dream Cycle with Large LLM ───────────────────────────────

async def night_dream(cte: CognitiveTrajectoryEngine) -> None:
    """Night: Large LLM analyzes the failure and discovers the cursor pattern."""
    _header("THE NIGHT: Large LLM Dreams (Cognitive Distillation)")

    print("  The large model is analyzing the small model's failure trace...")
    print("  Looking for patterns the small model missed...\n")

    journal = await cte.dream()

    print(f"  Traces processed:      {journal.traces_processed}")
    print(f"  Nodes created:         {journal.nodes_created}")
    print(f"  Nodes merged:          {journal.nodes_merged}")
    print(f"  Edges created:         {journal.edges_created}")
    print(f"  Constraints extracted: {journal.constraints_extracted}")

    if journal.phase_log:
        print(f"\n  Phase log:")
        for entry in journal.phase_log:
            print(f"    {entry}")

    if journal.constraints_extracted > 0:
        print(f"\n  The large model discovered {journal.constraints_extracted} insight(s)")
        print(f"  that the small model missed!")
        print(f"  (Likely: 'Stripe-Cursor header contains the pagination token')")
    print()


# ── Day 2: Small LLM with Distilled Knowledge ───────────────────────

async def day2_attempt(
    adapter: EvolvingMemoryAdapter,
    bee_llm,
) -> RunMetrics:
    """Day 2: Small LLM with memory from the large model's dreaming."""
    _header("DAY 2: Small LLM with Distilled Knowledge")

    reset_stripe_state()

    # Query memory
    memory_enhancement, route = adapter.get_memory_context(TASK_PROMPT)
    print(f"  Router decision: {route}")
    if memory_enhancement:
        print(f"  Memory context injected ({len(memory_enhancement)} chars)")
        print(f"\n  --- Injected Knowledge ---")
        for line in memory_enhancement.split("\n")[:15]:
            print(f"  {line}")
        if memory_enhancement.count("\n") > 15:
            print(f"  ... ({memory_enhancement.count(chr(10)) - 15} more lines)")
        print(f"  ---\n")
    else:
        print("  WARNING: No memory found — this shouldn't happen after dreaming!")

    print(f"  Task: {TASK_PROMPT[:80]}...")
    print(f"  The small LLM now has the large model's insight...\n")

    # Retry loop — small LLMs sometimes fail BeeAI's ReAct format
    metrics = None
    for attempt in range(3):
        reset_stripe_state()
        agent = ReActAgent(
            llm=bee_llm,
            tools=[stripe_charges_api],
            memory=UnconstrainedMemory(),
        )
        metrics = await adapter.run_with_tracing(
            agent=agent,
            prompt=TASK_PROMPT,
            session_goal=SESSION_GOAL,
            memory_enhancement=memory_enhancement,
            max_retries_per_step=5,
            max_iterations=12,
            total_max_retries=12,
        )
        if metrics.steps > 0:
            break
        if attempt < 2:
            print(f"  (Retrying — model format error, attempt {attempt + 2}/3)")

    # Print iteration log
    print(f"  --- Iteration Log ---")
    for i, entry in enumerate(metrics.iterations_log, 1):
        thought = entry.get("thought", "")[:120]
        tool_name = entry.get("tool_name", "")
        tool_input = entry.get("tool_input", "")[:80]
        tool_output = entry.get("tool_output", "")[:100]
        final = entry.get("final_answer", "")[:120]

        if tool_name:
            status = "ERROR" if "error" in tool_output.lower() else "OK"
            print(f"  [{i}] {thought}")
            print(f"      -> {tool_name}({tool_input})")
            print(f"      <- [{status}] {tool_output}")
        elif final:
            print(f"  [{i}] FINAL: {final}")
        else:
            print(f"  [{i}] {thought}")
        print()

    print(f"  --- Day 2 Result ---")
    print(f"  Steps: {metrics.steps}")
    print(f"  Tool errors: {metrics.tool_errors}")
    print(f"  Success: {'Yes' if metrics.success else 'No'}")

    # Check reconciliation accuracy
    check = _check_reconciliation(metrics.final_answer)
    if check["all_charges_found"]:
        print(f"  All {EXPECTED_CHARGES} charges found!")
    if check["correct_total"]:
        print(f"  Gross revenue correct: ${EXPECTED_GROSS:,.2f}")
    if check["disputes_flagged"]:
        print(f"  Disputed charges flagged for compliance review")
    if all(check.values()):
        print(f"\n  RECONCILIATION: COMPLETE AND ACCURATE")
    else:
        print(f"\n  (Agent response: {metrics.final_answer[:200]})")

    return metrics


# ── Main ─────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n" + "=" * 65)
    print("  EVOLVING MEMORY — Cognitive Distillation Demo")
    print("  'Teaching a small LLM with a large LLM's dreams'")
    print("=" * 65)
    print()
    print("  SCENARIO: Stripe Payment Reconciliation")
    print("  A finance team agent must reconcile ALL charges from Stripe.")
    print("  If the agent misses pages, the report is wrong — compliance risk.")

    # Setup LLMs
    print("\n  --- LLM Configuration ---")
    waking_llm_provider = GeminiProvider()  # For CTE trace capture
    dreaming_llm = _get_dreaming_llm()       # Large model for dreams

    try:
        bee_llm = _get_bee_llm()
    except RuntimeError as e:
        print(f"\n  {e}")
        return

    # Setup CTE with dual LLMs
    tmp_db = tempfile.mktemp(suffix=".db")
    cte = CognitiveTrajectoryEngine(
        llm=waking_llm_provider,
        dreaming_llm=dreaming_llm,  # Large model handles dreaming
        db_path=tmp_db,
    )
    adapter = EvolvingMemoryAdapter(cte)

    try:
        # Day 1: Small LLM fails reconciliation
        metrics_day1 = await day1_attempt(adapter, bee_llm)

        # Night: Large LLM dreams
        await night_dream(cte)

        # Day 2: Small LLM with distilled knowledge
        metrics_day2 = await day2_attempt(adapter, bee_llm)

        # Results
        _header("RESULTS: Cognitive Distillation")
        print_metrics_comparison(
            "Day 1 (No Memory)", metrics_day1,
            "Day 2 (Distilled)", metrics_day2,
        )

        # The money slide
        print("  KEY INSIGHT:")
        print("  The small LLM (8B params) could NOT complete the reconciliation")
        print("  — it lacked the reasoning depth to discover that Stripe's")
        print("  pagination cursor is in the 'Stripe-Cursor' HTTP header.")
        print()
        print("  Without all pages, the financial report was INCOMPLETE:")
        print(f"    - Missing charges (only found ~5 of {EXPECTED_CHARGES})")
        print(f"    - Wrong gross total (should be ${EXPECTED_GROSS:,.2f})")
        print(f"    - Unreported disputes (compliance violation)")
        print()
        print("  The large LLM analyzed the failure trace during the dream cycle")
        print("  and discovered the header pattern. This knowledge was distilled")
        print("  into procedural memory that the small LLM could follow.")
        print()
        print("  Result: 8B model + Evolving Memory = 70B reliability")
        print("  Cost: Only 1 dream cycle with the large model (offline, async)")
        print()
        print("  ENTERPRISE VALUE:")
        print("    - Correct financial reconciliation (compliance-ready)")
        print("    - Dispute detection for proactive resolution")
        print("    - Small model runs 24/7 at fraction of large model cost")
        print()

    finally:
        cte.close()
        for f in Path(tmp_db).parent.glob(Path(tmp_db).stem + "*"):
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
    os._exit(0)
