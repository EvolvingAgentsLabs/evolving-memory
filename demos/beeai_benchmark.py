#!/usr/bin/env python3
"""Demo 1: "The Trap" — BeeAI + Evolving Memory A/B Benchmark.

Demonstrates token savings and error reduction when a small LLM agent
uses evolving-memory to learn from its mistakes.

The Scenario:
  An agent must query a Spanish-named SQL database and calculate
  week-over-week growth. The database has traps:
    1. Column is `ingresos_centavos` (not `revenue`)
    2. Values are in centavos (must divide by 100)
    3. Python environment has no pandas

The Test:
  Attempt 1 — Agent runs without memory (baseline). It stumbles on column
  names, wrong Python imports, and burns tokens correcting itself.

  Dream Cycle — Evolving-memory consolidates the failure traces, extracts
  negative constraints and a working strategy.

  Attempt 2 — Same agent, same prompt, but with memory-injected guidance.
  It succeeds cleanly in fewer steps with fewer tokens.

Usage:
    GEMINI_API_KEY=... PYTHONPATH=src python demos/beeai_benchmark.py

    # Or with Ollama (local):
    BEE_LLM=ollama:llama3.1:8b PYTHONPATH=src python demos/beeai_benchmark.py

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
from mock_tools import sales_db_query, python_calc


# ── Configuration ────────────────────────────────────────────────────

TASK_PROMPT = (
    "Query the sales database for the 'Electronica' category in Q3. "
    "Get the weekly revenue data, then calculate the week-over-week "
    "percentage growth for each week. Show the results as a formatted table."
)

SESSION_GOAL = "Analyze Q3 Electronica sales with week-over-week growth"


def _header(title: str) -> None:
    width = 65
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def _get_bee_llm():
    """Get the BeeAI ChatModel from environment or defaults."""
    model_name = os.environ.get("BEE_LLM", "")
    if model_name:
        print(f"  LLM: {model_name}")
        return ChatModel.from_name(model_name)

    # Default: use Gemini via LiteLLM if GEMINI_API_KEY is set
    if os.environ.get("GEMINI_API_KEY"):
        model = os.environ.get("BEE_GEMINI_MODEL", "gemini-2.5-flash-lite")
        print(f"  LLM: gemini/{model}")
        return GeminiChatModel(model)

    # Fallback: try Ollama local models
    for name in ["ollama:llama3.2", "ollama:llama3.1:8b", "ollama:llama3.1", "groq:llama-3.1-8b-instant"]:
        try:
            llm = ChatModel.from_name(name)
            print(f"  LLM: {name}")
            return llm
        except Exception:
            continue

    raise RuntimeError(
        "No LLM available. Set GEMINI_API_KEY or BEE_LLM env var."
    )


# ── Attempt Runner ───────────────────────────────────────────────────

async def run_attempt(
    adapter: EvolvingMemoryAdapter,
    bee_llm,
    label: str,
    with_memory: bool = False,
    agent_retries: int = 3,
) -> RunMetrics:
    """Run the BeeAI agent with or without memory guidance."""
    _header(f"{label}")

    # Get memory context if enabled
    memory_enhancement = ""
    route = "zero_shot"
    if with_memory:
        memory_enhancement, route = adapter.get_memory_context(TASK_PROMPT)
        print(f"  Router decision: {route}")
        if memory_enhancement:
            print(f"  Memory context injected ({len(memory_enhancement)} chars)")
            print(f"  Preview: {memory_enhancement[:200]}...")
        else:
            print("  No relevant memory found — running from scratch")
        print()

    print(f"  Task: {TASK_PROMPT[:80]}...")
    print(f"  Running agent...\n")

    # Retry loop — small LLMs sometimes fail BeeAI's ReAct format
    metrics = None
    for attempt in range(agent_retries):
        agent = ReActAgent(
            llm=bee_llm,
            tools=[sales_db_query, python_calc],
            memory=UnconstrainedMemory(),
        )
        metrics = await adapter.run_with_tracing(
            agent=agent,
            prompt=TASK_PROMPT,
            session_goal=SESSION_GOAL,
            memory_enhancement=memory_enhancement,
            max_retries_per_step=5,
            max_iterations=15,
            total_max_retries=12,
        )
        if metrics.steps > 0:
            break
        if attempt < agent_retries - 1:
            print(f"  (Retrying — model format error, attempt {attempt + 2}/{agent_retries})")

    # Print iteration log
    print(f"  --- Iteration Log ---")
    for i, entry in enumerate(metrics.iterations_log, 1):
        thought = entry.get("thought", "")[:100]
        tool_name = entry.get("tool_name", "")
        tool_output = entry.get("tool_output", "")[:80]
        final = entry.get("final_answer", "")[:100]

        if tool_name:
            status = "ERROR" if any(e in tool_output.lower() for e in ["error", "traceback"]) else "OK"
            print(f"  [{i}] Thought: {thought}")
            print(f"      Tool: {tool_name} -> [{status}] {tool_output}")
        elif final:
            print(f"  [{i}] Final Answer: {final}")
        else:
            print(f"  [{i}] Thought: {thought}")
        print()

    # Summary
    print(f"  --- Summary ---")
    print(f"  Steps: {metrics.steps}")
    print(f"  Tool calls: {metrics.tool_calls} ({metrics.tool_errors} errors)")
    print(f"  Tokens (est): {metrics.total_tokens}")
    print(f"  Latency: {metrics.latency_s:.1f}s")
    print(f"  Success: {'Yes' if metrics.success else 'No'}")
    if not metrics.success:
        print(f"  Error: {metrics.final_answer[:200]}")

    return metrics


# ── Main ─────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n" + "=" * 65)
    print("  EVOLVING MEMORY — BeeAI Benchmark: 'The Trap'")
    print("  Demonstrating token savings via procedural memory")
    print("=" * 65)

    # Setup
    gemini = GeminiProvider()
    tmp_db = tempfile.mktemp(suffix=".db")
    cte = CognitiveTrajectoryEngine(llm=gemini, db_path=tmp_db)
    adapter = EvolvingMemoryAdapter(cte)

    try:
        bee_llm = _get_bee_llm()
    except RuntimeError as e:
        print(f"\n  {e}")
        cte.close()
        return

    try:
        # ── Attempt 1: Baseline (no memory) ──
        metrics_baseline = await run_attempt(
            adapter, bee_llm,
            label="ATTEMPT 1: Baseline (No Memory)",
            with_memory=False,
        )

        # ── Dream Cycle ──
        _header("DREAM CYCLE")
        print("  Consolidating traces into procedural memory...\n")
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

        # ── Attempt 2: With Memory ──
        metrics_memory = await run_attempt(
            adapter, bee_llm,
            label="ATTEMPT 2: With Evolving Memory",
            with_memory=True,
        )

        # ── Results Comparison ──
        _header("RESULTS: A/B Comparison")
        print_metrics_comparison(
            "Without Memory", metrics_baseline,
            "With Memory", metrics_memory,
        )

        # Money slide callout
        if metrics_memory.total_tokens < metrics_baseline.total_tokens:
            savings_pct = (1 - metrics_memory.total_tokens / max(metrics_baseline.total_tokens, 1)) * 100
            print(f"  Token savings: {savings_pct:.0f}%")
            print(f"  'Evolving Memory makes a small LLM as reliable as a large one'")
        print()

    finally:
        cte.close()
        # Clean up temp files
        for f in Path(tmp_db).parent.glob(Path(tmp_db).stem + "*"):
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
    os._exit(0)
