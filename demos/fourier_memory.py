#!/usr/bin/env python3
"""Demo: Fourier Transform Reasoning with Memory Consolidation.

Shows how evolving-memory's CTE accumulates mathematical reasoning traces,
consolidates them through dreaming, then uses recalled strategies to
answer related questions and solve new problems.

Usage:
    GEMINI_API_KEY=... PYTHONPATH=src python demos/fourier_memory.py
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
    RouterPath,
)
from evolving_memory.models.hierarchy import TraceOutcome
from evolving_memory.llm.gemini_provider import GeminiProvider


# ── Helpers ───────────────────────────────────────────────────────────

def _header(title: str) -> None:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}\n")


def _sub(label: str, text: str, max_len: int = 250) -> None:
    truncated = text.strip().replace("\n", " ")[:max_len]
    print(f"  → {label}: {truncated}")


# ── Phase 1: Learn Fourier Transform ─────────────────────────────────

async def phase1_learn(
    cte: CognitiveTrajectoryEngine,
    gemini: GeminiProvider,
) -> None:
    _header("PHASE 1: LEARNING FOURIER TRANSFORM")

    # Trace 1: Understand the Fourier Transform (GOAL level)
    print("  [Trace 1] Understanding the Fourier Transform...\n")
    with cte.session("learn Fourier Transform") as logger:
        with logger.trace(
            HierarchyLevel.GOAL,
            "Understand the Fourier Transform",
            tags=["fourier", "math", "learning"],
        ) as ctx:
            # Action 1: Intuition
            resp1 = await gemini.complete(
                "Explain the intuition behind the Fourier Transform in 3-4 sentences. "
                "Focus on why it's useful and what it does conceptually.",
                system="You are a math tutor. Be clear and concise.",
            )
            ctx.action(
                reasoning="Build intuitive understanding of Fourier Transform",
                action_payload="Ask: explain intuition behind Fourier Transform",
                result=resp1,
            )
            _sub("Intuition", resp1)

            # Action 2: Mathematical definition
            resp2 = await gemini.complete(
                "What is the mathematical definition of the Discrete Fourier Transform (DFT)? "
                "Give the formula and explain each variable.",
                system="You are a math tutor. Be precise with notation.",
            )
            ctx.action(
                reasoning="Learn the formal mathematical definition of DFT",
                action_payload="Ask: mathematical definition of DFT",
                result=resp2,
            )
            _sub("DFT Definition", resp2)

            # Action 3: Concrete example
            resp3 = await gemini.complete(
                "Give a concrete example: how would you decompose a square wave "
                "into its Fourier components? Describe the first 3 harmonics.",
                system="You are a math tutor. Use specific numbers.",
            )
            ctx.action(
                reasoning="Ground understanding with a concrete square wave example",
                action_payload="Ask: decompose square wave into Fourier components",
                result=resp3,
            )
            _sub("Square Wave Example", resp3)
            ctx.set_outcome(TraceOutcome.SUCCESS, confidence=0.90)

        # Trace 2: Apply FFT to signal processing (TACTICAL level)
        print(f"\n  [Trace 2] Applying FFT to signal processing...\n")
        with logger.trace(
            HierarchyLevel.TACTICAL,
            "Apply FFT to signal processing",
            tags=["fft", "signal_processing", "application"],
        ) as ctx:
            # Action 1: FFT vs DFT
            resp4 = await gemini.complete(
                "How does FFT differ from DFT? What makes FFT faster?",
                system="You are a computer science tutor. Be precise.",
            )
            ctx.action(
                reasoning="Understand the computational advantage of FFT over DFT",
                action_payload="Ask: FFT vs DFT differences",
                result=resp4,
            )
            _sub("FFT vs DFT", resp4)

            # Action 2: Computational complexity
            resp5 = await gemini.complete(
                "What are the computational complexities of DFT and FFT? "
                "Express in Big-O notation and explain why.",
                system="You are a computer science tutor. Be precise.",
            )
            ctx.action(
                reasoning="Learn the computational complexities to choose the right algorithm",
                action_payload="Ask: computational complexities of DFT vs FFT",
                result=resp5,
            )
            _sub("Complexity", resp5)

            # Action 3: Practical usage
            resp6 = await gemini.complete(
                "When would you use FFT vs DFT in practice? "
                "Give 2-3 real-world scenarios for each.",
                system="You are a signal processing expert. Be practical.",
            )
            ctx.action(
                reasoning="Understand practical scenarios for choosing FFT vs DFT",
                action_payload="Ask: practical scenarios for FFT vs DFT",
                result=resp6,
            )
            _sub("Practical Use", resp6)
            ctx.set_outcome(TraceOutcome.SUCCESS, confidence=0.85)

    print(f"\n  Stored 2 traces (6 total actions)")


# ── Phase 2: Dream ───────────────────────────────────────────────────

async def phase2_dream(cte: CognitiveTrajectoryEngine) -> None:
    _header("PHASE 2: DREAMING")

    journal = await cte.dream()

    print(f"  Traces processed:      {journal.traces_processed}")
    print(f"  Nodes created:         {journal.nodes_created}")
    print(f"  Nodes merged:          {journal.nodes_merged}")
    print(f"  Edges created:         {journal.edges_created}")
    print(f"  Constraints extracted: {journal.constraints_extracted}")
    if journal.phase_log:
        print(f"\n  Phase log:")
        for entry in journal.phase_log:
            print(f"    • {entry}")


# ── Phase 3: Query Memory ────────────────────────────────────────────

async def phase3_query(
    cte: CognitiveTrajectoryEngine,
    gemini: GeminiProvider,
) -> None:
    _header("PHASE 3: QUERYING MEMORY")

    # Query 1: Should find Fourier memory
    print("  ── Query 1: Related topic ──")
    print('  "How to analyze frequency components of a signal?"')
    decision1 = cte.query("How to analyze frequency components of a signal?")
    print(f"    Route:      {decision1.path.value}")
    print(f"    Confidence: {decision1.confidence:.2f}")
    print(f"    Reasoning:  {decision1.reasoning[:120]}")

    if decision1.path == RouterPath.MEMORY_TRAVERSAL and decision1.entry_point:
        ep = decision1.entry_point
        print(f"\n    Entry point: {ep.parent_node.goal}")
        print(f"    Similarity:  {ep.similarity_score:.2f}")
        print(f"    Composite:   {ep.composite_score:.2f}")

        # Traverse recalled strategy
        state = cte.begin_traversal(ep)
        recalled_steps = []
        while True:
            child, state = cte.next_step(state)
            if child is None:
                break
            recalled_steps.append(child)
            print(f"    Step {child.step_index}: {child.action[:80]}")

        # Use recalled memory to solve a new problem
        memory_context = f"You previously learned about: {ep.parent_node.goal}\n"
        for s in recalled_steps:
            memory_context += f"- {s.reasoning}: {s.result[:150]}\n"

        print(f"\n  Using recalled memory to solve a NEW problem...")
        new_answer = await gemini.complete(
            f"Using your prior knowledge:\n{memory_context}\n\n"
            f"Now solve this new problem: A signal contains frequencies at 50 Hz, "
            f"120 Hz, and 300 Hz, sampled at 1000 Hz. How many samples do you need "
            f"for the FFT to resolve all three frequencies? What frequency resolution "
            f"would you get with 1024 samples?",
            system="You are a signal processing expert. Use your prior knowledge to solve this.",
        )
        _sub("New Problem Solution", new_answer, max_len=400)
    else:
        print("    (No memory traversal — zero-shot)")

    # Query 2: Unrelated topic — should get ZERO_SHOT
    print(f"\n\n  ── Query 2: Unrelated topic ──")
    print('  "What is the Laplace Transform?"')
    decision2 = cte.query("What is the Laplace Transform?")
    print(f"    Route:      {decision2.path.value}")
    print(f"    Confidence: {decision2.confidence:.2f}")
    print(f"    Reasoning:  {decision2.reasoning[:120]}")

    if decision2.path == RouterPath.ZERO_SHOT:
        print(f"    ✓ Correctly routed to ZERO_SHOT (no Laplace memory exists)")
    elif decision2.path == RouterPath.MEMORY_TRAVERSAL:
        print(f"    ~ Routed to MEMORY_TRAVERSAL (Laplace is related to Fourier)")
        if decision2.entry_point:
            print(f"    Entry point: {decision2.entry_point.parent_node.goal}")


# ── Main ──────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n🧠 Evolving Memory — Fourier Transform Reasoning Demo")
    print("=" * 55)

    # Setup
    gemini = GeminiProvider()
    tmp = tempfile.mktemp(suffix=".db")
    cte = CognitiveTrajectoryEngine(llm=gemini, db_path=tmp)

    try:
        await phase1_learn(cte, gemini)
        await phase2_dream(cte)
        await phase3_query(cte, gemini)

        _header("DONE")
        print("  Demo complete. Mathematical reasoning accumulated in memory.")
        print(f"  DB: {tmp}")
    finally:
        cte.close()
        # Clean up temp DB
        for f in Path(tmp).parent.glob(Path(tmp).stem + "*"):
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
    # Avoid segfault during interpreter shutdown (faiss/torch cleanup race)
    os._exit(0)
