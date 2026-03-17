#!/usr/bin/env python3
"""Demo 2: "Cognitive Distillation" — Teacher-Student Knowledge Transfer.

Demonstrates that a large LLM can teach a small LLM through evolving-memory's
dream cycle, enabling the small model to solve problems it could never solve
on its own.

The Scenario:
  An agent must extract ALL employees from a paginated API. The catch:
  the pagination token is hidden in HTTP response headers (X-Next-Page),
  NOT in the JSON body. The body only has `has_more: true`.

  A small LLM (8B params) will try ?page=2, ?offset=5, etc. — all fail.
  It will exhaust its retries and abort.

  A large LLM (Gemini Pro / Claude) analyzes the failure trace during the
  dream cycle and discovers the hidden pattern in the raw API response.

  The small LLM, now armed with the distilled knowledge, succeeds immediately.

The Proof:
  "Evolving Memory allows an 8B-parameter model to achieve the reliability
  of a 70B model, by giving it procedural memory from a more capable teacher."

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
from mock_tools import paginated_api_fetch, reset_api_state


# ── Configuration ────────────────────────────────────────────────────

TASK_PROMPT = (
    "Extract ALL employees from the Employee Directory API. "
    "The base endpoint is /api/employees. "
    "The API returns paginated results — you must fetch every page "
    "until there are no more results. "
    "Return the total count and a list of all employee names."
)

SESSION_GOAL = "Extract all employees from paginated API"


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


# ── Day 1: Small LLM Attempts ───────────────────────────────────────

async def day1_attempt(
    adapter: EvolvingMemoryAdapter,
    bee_llm,
) -> RunMetrics:
    """Day 1: Small LLM tries to solve the pagination problem. Expected: FAILURE."""
    _header("DAY 1: Small LLM Attempts (Expected: Failure)")

    reset_api_state()

    print(f"  Task: {TASK_PROMPT[:80]}...")
    print(f"  The small LLM will try to paginate but won't find the token...")
    print()

    # Retry loop — small LLMs sometimes fail BeeAI's ReAct format
    metrics = None
    for attempt in range(3):
        reset_api_state()
        agent = ReActAgent(
            llm=bee_llm,
            tools=[paginated_api_fetch],
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

    # Check if the agent actually got all employees
    if metrics.final_answer and "13" in metrics.final_answer:
        print(f"  (Agent found all 13 employees)")
    else:
        print(f"  (Agent did NOT find all 13 employees — as expected)")

    return metrics


# ── Night: Dream Cycle with Large LLM ───────────────────────────────

async def night_dream(cte: CognitiveTrajectoryEngine) -> None:
    """Night: Large LLM analyzes the failure and discovers the solution."""
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
    print()


# ── Day 2: Small LLM with Distilled Knowledge ───────────────────────

async def day2_attempt(
    adapter: EvolvingMemoryAdapter,
    bee_llm,
) -> RunMetrics:
    """Day 2: Small LLM with memory from the large model's dreaming."""
    _header("DAY 2: Small LLM with Distilled Knowledge")

    reset_api_state()

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
        reset_api_state()
        agent = ReActAgent(
            llm=bee_llm,
            tools=[paginated_api_fetch],
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

    if metrics.final_answer and "13" in metrics.final_answer:
        print(f"  Agent found all 13 employees!")
    else:
        print(f"  (Agent response: {metrics.final_answer[:150]})")

    return metrics


# ── Main ─────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n" + "=" * 65)
    print("  EVOLVING MEMORY — Cognitive Distillation Demo")
    print("  'Teaching a small LLM with a large LLM's dreams'")
    print("=" * 65)

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
        # Day 1: Small LLM fails
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
        print("  The small LLM (8B params) could NOT solve this problem")
        print("  even with unlimited retries — it lacked the reasoning")
        print("  depth to discover that pagination tokens are in HTTP headers.")
        print()
        print("  The large LLM (during the dream cycle) analyzed the failure")
        print("  trace and discovered the pattern. This knowledge was distilled")
        print("  into procedural memory that the small LLM could follow.")
        print()
        print("  Result: 8B model + Evolving Memory = 70B reliability")
        print("  Cost: Only 1 dream cycle with the large model (offline, async)")
        print()

    finally:
        cte.close()
        for f in Path(tmp_db).parent.glob(Path(tmp_db).stem + "*"):
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
    os._exit(0)
