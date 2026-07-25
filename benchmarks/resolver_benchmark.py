#!/usr/bin/env python3
"""Does indexing what a component is *for* beat indexing what it *is*?

Description-matching is what the ecosystem does today: embed a component's
description, embed the query, rank by similarity. This measures the alternative
on the case that separates them — a task phrased in the vocabulary of the job
rather than of the implementation.

Run:
    GEMINI_API_KEY=... python benchmarks/resolver_benchmark.py

Without a key it uses the deterministic template writer, which is a weaker
applicability text and will understate the difference. That is the honest floor,
not the headline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evolving_memory.resolver import ApplicabilityWriter, Component, DualIndex  # noqa: E402

# ── Corpus ────────────────────────────────────────────────────────────────────
# Descriptions are written the way a library author writes them: about the
# implementation. Tasks are written the way someone with a problem writes them.

COMPONENTS = [
    Component(id="cable", name="Conductor ampacity tables", domain="electrical", kind="tool",
              description="Parses IEC 60364 conductor ampacity and derating tables."),
    Component(id="rcd", name="RCD trip curve calculator", domain="electrical", kind="tool",
              description="Computes residual-current device trip characteristics."),
    Component(id="invoice", name="Invoice line extractor", domain="finance", kind="tool",
              description="Extracts line items and totals from PDF invoices."),
    Component(id="ledger", name="Double-entry reconciler", domain="finance", kind="agent",
              description="Reconciles debit and credit entries across ledgers."),
    Component(id="retry", name="Exponential backoff wrapper", domain="infra", kind="tool",
              description="Wraps callables with exponential backoff and jitter."),
    Component(id="tracer", name="Span emitter", domain="infra", kind="tool",
              description="Emits OpenTelemetry spans around function calls."),
    Component(id="rota", name="Shift rota generator", domain="ops", kind="agent",
              description="Generates staff rotas subject to availability constraints."),
    Component(id="triage", name="Symptom severity classifier", domain="health", kind="agent",
              description="Classifies reported symptoms into severity bands."),
    Component(id="dose", name="Weight-based dosing table", domain="health", kind="tool",
              description="Looks up paediatric dosing by patient weight."),
    Component(id="chunker", name="Recursive text splitter", domain="nlp", kind="tool",
              description="Splits documents into overlapping chunks by separator."),
]

#: (task as a user would phrase it, the component that should win)
TASKS = [
    ("check whether the wiring in this flat is safe before we sign the lease", "cable"),
    ("someone keeps getting shocks off the washing machine, what protects them", "rcd"),
    ("we paid this supplier twice and need to prove it from the paperwork", "invoice"),
    ("the books do not balance at month end", "ledger"),
    ("the API keeps failing under load and we hammer it harder each time", "retry"),
    ("we cannot tell which service is slow in production", "tracer"),
    ("nobody knows who is working the bank holiday", "rota"),
    ("a parent is describing symptoms and we need to know how urgent it is", "triage"),
    ("how much of this do we give a child who weighs 14 kilos", "dose"),
    ("the document is too long to fit in the context window", "chunker"),
]


def evaluate(index: DualIndex, task_weight: float) -> tuple[float, float, list[str]]:
    """Returns (accuracy@1, MRR, misses)."""
    hits = 0
    reciprocal = 0.0
    misses = []
    for task, expected in TASKS:
        ranked = index.resolve(task, task_weight=task_weight, top_k=len(COMPONENTS), use_boost=False)
        ids = [m.component.id for m in ranked]
        if ids and ids[0] == expected:
            hits += 1
        else:
            misses.append(f"{task[:44]}… → {ids[0] if ids else '∅'} (wanted {expected})")
        if expected in ids:
            reciprocal += 1.0 / (ids.index(expected) + 1)
    n = len(TASKS)
    return hits / n, reciprocal / n, misses


def main() -> int:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("No GEMINI_API_KEY — cannot embed. Set it and re-run.")
        return 1

    from evolving_memory.embeddings.encoder import EmbeddingEncoder
    from evolving_memory.llm.gemini_provider import GeminiProvider

    encoder = EmbeddingEncoder(api_key=key)

    try:
        writer = ApplicabilityWriter(_Adapter(GeminiProvider(api_key=key)))
        mode = "LLM-written applicability"
    except Exception as exc:  # noqa: BLE001
        print(f"  (falling back to template applicability: {exc})")
        writer = ApplicabilityWriter()
        mode = "template applicability (floor, not headline)"

    print(f"Indexing {len(COMPONENTS)} components — {mode}")
    index = DualIndex(encoder, writer)
    for component in COMPONENTS:
        index.add(component)

    print(f"\n{len(TASKS)} tasks, each phrased as a problem rather than a capability.\n")
    print(f"{'task_weight':>12}  {'what it means':<34}  {'acc@1':>6}  {'MRR':>6}")
    print("-" * 68)

    results = {}
    for weight, label in ((0.0, "description-matching (the baseline)"),
                          (0.5, "both, evenly"),
                          (1.0, "applicability only")):
        acc, mrr, misses = evaluate(index, weight)
        results[weight] = (acc, mrr, misses)
        print(f"{weight:>12.1f}  {label:<34}  {acc:>6.0%}  {mrr:>6.3f}")

    base_acc = results[0.0][0]
    best_weight = max(results, key=lambda w: (results[w][0], results[w][1]))
    best_acc = results[best_weight][0]

    print()
    if best_acc > base_acc:
        print(f"Task-aware retrieval wins: {base_acc:.0%} → {best_acc:.0%} at "
              f"task_weight={best_weight}.")
    elif best_acc == base_acc:
        print(f"No difference on this corpus ({base_acc:.0%} both ways). "
              f"That is a real result — report it.")
    else:
        print(f"Description-matching wins here ({base_acc:.0%} vs {best_acc:.0%}). "
              f"Report that too.")

    if results[best_weight][2]:
        print("\nStill missed at the best setting:")
        for miss in results[best_weight][2]:
            print(f"  {miss}")
    return 0


class _Adapter:
    """Bridges a provider's async/typed interface to `generate(prompt) -> str`."""

    def __init__(self, provider) -> None:
        self._p = provider

    def generate(self, prompt: str) -> str:
        import asyncio

        # Providers expose async `complete`; the writer wants a sync `generate`.
        return asyncio.run(self._p.complete(prompt)).strip()


if __name__ == "__main__":
    raise SystemExit(main())
