#!/usr/bin/env python3
"""Demo: Recursive Image Analysis with Memory Consolidation.

Shows how evolving-memory's CTE accumulates visual analysis traces,
consolidates them through dreaming, then uses recalled memory to
enrich a second encounter with the same image.

Usage:
    GEMINI_API_KEY=... PYTHONPATH=src python demos/recursive_image_analysis.py
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import urllib.request
from pathlib import Path

# Prevent faiss/torch BLAS threading segfault on some platforms
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from evolving_memory import (
    CognitiveTrajectoryEngine,
    HierarchyLevel,
    RouterPath,
)
from evolving_memory.models.hierarchy import TraceOutcome, TraceSource
from evolving_memory.llm.gemini_provider import GeminiProvider

# ── Config ────────────────────────────────────────────────────────────

# Complex kitchen scene (CC0 / Unsplash)
IMAGE_URL = "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=800&q=80"


# ── Helpers ───────────────────────────────────────────────────────────

def _header(title: str) -> None:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}\n")


def _sub(label: str, text: str) -> None:
    # Indent continuation lines
    lines = text.strip().split("\n")
    print(f"  → {label}: {lines[0]}")
    for line in lines[1:]:
        print(f"    {line}")


def download_image(url: str) -> str:
    """Download an image and return its base64 encoding."""
    print(f"Downloading image from {url[:60]}...")
    req = urllib.request.Request(url, headers={"User-Agent": "evolving-memory-demo/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    b64 = base64.b64encode(data).decode()
    print(f"  Downloaded {len(data):,} bytes → {len(b64):,} chars base64\n")
    return b64


# ── Phase 1: First Encounter ─────────────────────────────────────────

async def phase1_first_encounter(
    cte: CognitiveTrajectoryEngine,
    gemini: GeminiProvider,
    image_b64: str,
) -> None:
    _header("PHASE 1: FIRST ENCOUNTER (Zero-Shot)")

    with cte.session("analyze kitchen scene") as logger:
        # Step 1: whole-scene analysis
        with logger.trace(HierarchyLevel.GOAL, "Analyze kitchen scene") as ctx:
            scene_desc = await gemini.complete_vision(
                "Describe this scene in detail. List the main objects you see.",
                image_b64,
                system="You are a precise visual analyst. Be concise but thorough.",
            )
            ctx.action(
                reasoning="Analyze entire scene to identify key objects",
                action_payload="complete_vision(scene overview)",
                result=scene_desc,
            )
            print(f"[Scene Analysis]\n  {scene_desc[:300]}...\n")

            # Extract objects from the description for deeper analysis
            objects_resp = await gemini.complete_json(
                f"Given this scene description, list the 3 most interesting objects "
                f"or areas to analyze further. Return JSON: "
                f'{{"objects": ["object1", "object2", "object3"]}}\n\n'
                f"Description: {scene_desc}",
                system="Return valid JSON only.",
            )
            objects = objects_resp.data.get("objects", ["main area", "background", "foreground"])
            ctx.action(
                reasoning="Extract key objects from scene for deeper analysis",
                action_payload="complete_json(extract objects)",
                result=f"Objects identified: {', '.join(objects)}",
            )
            ctx.set_outcome(TraceOutcome.SUCCESS, confidence=0.85)

        # Step 2: detailed analysis of each detected object
        for i, obj in enumerate(objects, 1):
            with logger.trace(
                HierarchyLevel.TACTICAL,
                f"Detailed analysis of: {obj}",
                tags=["detail", obj.replace(" ", "_")],
            ) as ctx:
                detail = await gemini.complete_vision(
                    f"Focus specifically on the '{obj}' in this image. "
                    f"Describe it in detail: what it looks like, its condition, "
                    f"any notable characteristics.",
                    image_b64,
                    system="You are a precise visual analyst. Focus only on the requested object/area.",
                )
                ctx.action(
                    reasoning=f"Zoom into '{obj}' for detailed analysis",
                    action_payload=f"complete_vision(detail: {obj})",
                    result=detail,
                )
                # Summarize findings as a second action
                summary = await gemini.complete(
                    f"In one sentence, what is the most notable aspect of: {detail[:500]}",
                    system="Be concise. One sentence only.",
                )
                ctx.action(
                    reasoning=f"Summarize key finding about '{obj}'",
                    action_payload=f"complete(summarize {obj})",
                    result=summary,
                )
                ctx.set_outcome(TraceOutcome.SUCCESS, confidence=0.80)
                _sub(f"Object {i}: {obj}", detail[:200])

    trace_count = len(logger.traces)
    print(f"\n  Stored {trace_count} traces (1 parent + {trace_count - 1} detailed)")


# ── Phase 2: Dream Consolidation ─────────────────────────────────────

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


# ── Phase 3: Second Encounter ────────────────────────────────────────

async def phase3_second_encounter(
    cte: CognitiveTrajectoryEngine,
    gemini: GeminiProvider,
    image_b64: str,
) -> None:
    _header("PHASE 3: SECOND ENCOUNTER (Memory-Augmented)")

    # Query memory for related experiences
    decision = cte.query("kitchen scene analysis")
    print(f"  Memory query: \"kitchen scene analysis\"")
    print(f"    Route:      {decision.path.value}")
    print(f"    Confidence: {decision.confidence:.2f}")
    print(f"    Reasoning:  {decision.reasoning[:120]}")

    memory_context = ""

    if decision.path == RouterPath.MEMORY_TRAVERSAL and decision.entry_point:
        ep = decision.entry_point
        print(f"\n  Entry point: {ep.parent_node.goal}")
        print(f"    Similarity: {ep.similarity_score:.2f}")
        print(f"    Composite:  {ep.composite_score:.2f}")

        # Traverse the stored strategy
        state = cte.begin_traversal(ep)
        steps = []
        while True:
            child, state = cte.next_step(state)
            if child is None:
                break
            steps.append(child)
            print(f"    Step {child.step_index}: {child.action[:80]}")
            if child.result:
                print(f"      Result: {child.result[:100]}...")

        # Build memory context for augmented analysis
        memory_context = (
            f"PRIOR ANALYSIS (from memory):\n"
            f"Goal: {ep.parent_node.goal}\n"
        )
        for s in steps:
            memory_context += f"- Step {s.step_index}: {s.reasoning} → {s.result[:150]}\n"
        if ep.parent_node.negative_constraints:
            memory_context += f"Constraints: {', '.join(ep.parent_node.negative_constraints)}\n"
    else:
        print(f"\n  No prior memory found — analyzing fresh.")

    # Now do the augmented analysis
    print()
    if memory_context:
        augmented_prompt = (
            f"You have prior knowledge about this scene from a previous analysis:\n\n"
            f"{memory_context}\n\n"
            f"Now re-analyze this image. Build on your prior knowledge — "
            f"confirm, correct, or add new observations. "
            f"What did the previous analysis miss? What can you add?"
        )
    else:
        augmented_prompt = "Describe this scene in detail. List the main objects you see."

    augmented_desc = await gemini.complete_vision(
        augmented_prompt,
        image_b64,
        system="You are a precise visual analyst with access to prior analysis memory.",
    )
    print(f"[Memory-Augmented Analysis]\n  {augmented_desc[:500]}")

    # Store this second encounter too
    with cte.session("re-analyze kitchen scene") as logger:
        with logger.trace(
            HierarchyLevel.GOAL,
            "Memory-augmented kitchen scene re-analysis",
            tags=["memory_augmented", "second_encounter"],
        ) as ctx:
            ctx.action(
                reasoning="Re-analyze with memory context from prior encounter",
                action_payload="complete_vision(augmented analysis)",
                result=augmented_desc,
            )
            ctx.set_outcome(TraceOutcome.SUCCESS, confidence=0.92)


# ── Main ──────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n🧠 Evolving Memory — Recursive Image Analysis Demo")
    print("=" * 50)

    # Setup
    gemini = GeminiProvider()
    tmp = tempfile.mktemp(suffix=".db")
    cte = CognitiveTrajectoryEngine(llm=gemini, db_path=tmp)

    try:
        image_b64 = download_image(IMAGE_URL)
        await phase1_first_encounter(cte, gemini, image_b64)
        await phase2_dream(cte)
        await phase3_second_encounter(cte, gemini, image_b64)

        _header("DONE")
        print("  Demo complete. Memory accumulated across encounters.")
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
