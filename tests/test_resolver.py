"""Tests for task-aware dual-embedding resolution.

The central claim is falsifiable and gets its own test: for a task phrased in
the vocabulary of the job rather than of the implementation, indexing what a
component is *for* retrieves the right component where indexing what it *is*
does not. If that stops holding, the module has no reason to exist.

No network. A deterministic bag-of-words encoder stands in for the real one, so
these measure the retrieval maths rather than an embedding model's mood.
"""

from __future__ import annotations

import numpy as np
import pytest

from evolving_memory.resolver import (
    ApplicabilityWriter,
    Component,
    DualIndex,
    ensure_applicability,
)


# ── Test doubles ──────────────────────────────────────────────────────────────


class BagOfWordsEncoder:
    """Deterministic hashed bag-of-words. Shared vocabulary → similar vectors."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for word in str(text).lower().split():
            word = word.strip(".,:;()[]\"'")
            if word:
                vec[hash(word) % self._dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


class ListIndex:
    """Exhaustive in-memory index, so tests need no faiss."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: dict[str, np.ndarray] = {}

    def add(self, node_id: str, vector) -> None:
        self._vectors[node_id] = np.asarray(vector, dtype=np.float32)

    def remove(self, node_id: str) -> None:
        self._vectors.pop(node_id, None)

    def search(self, query_vector, top_k: int = 5):
        q = np.asarray(query_vector, dtype=np.float32)
        scored = []
        for nid, v in self._vectors.items():
            denom = float(np.linalg.norm(q) * np.linalg.norm(v))
            scored.append((nid, float(np.dot(q, v) / denom) if denom else 0.0))
        scored.sort(key=lambda t: -t[1])
        return scored[:top_k]


class ScriptedLLM:
    """Returns canned applicability text, and records what it was asked."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        for key, reply in self.replies.items():
            if key in prompt:
                return reply
        return "generic applicability"


@pytest.fixture
def index():
    return DualIndex(BagOfWordsEncoder(), index_factory=ListIndex)


# ── The claim ─────────────────────────────────────────────────────────────────


def test_applicability_finds_what_content_alone_misses():
    """A task phrased as a job, not as an implementation.

    The component's own words are about IEC tables and conductors. The task is
    about a wiring inspection. They share no vocabulary, so a content-only index
    ranks the wrong thing first. The applicability text bridges them.
    """
    llm = ScriptedLLM({
        "Cable sizing": "Use this when checking whether wiring in a building is "
                        "safe and up to code during an inspection.",
        "Invoice": "Use this when extracting totals from billing documents.",
    })
    idx = DualIndex(BagOfWordsEncoder(), ApplicabilityWriter(llm), index_factory=ListIndex)

    idx.add(Component(
        id="cable", name="Cable sizing tables",
        description="Parses IEC 60364 conductor ampacity tables",
    ))
    idx.add(Component(
        id="invoice", name="Invoice inspection parser",
        description="Inspects invoice documents and checks their wiring of line items",
    ))

    task = "check whether the wiring in this building is safe and up to code"

    # Content only: the decoy wins, because it shares words with the task.
    content_only = idx.resolve(task, task_weight=0.0, top_k=2)
    assert content_only[0].component.id == "invoice"

    # Applicability: the right component wins.
    task_aware = idx.resolve(task, task_weight=1.0, top_k=2)
    assert task_aware[0].component.id == "cable"


def test_task_weight_moves_the_ranking_monotonically(index):
    llm = ScriptedLLM({"Alpha": "billing and invoices", "Beta": "wiring and cables"})
    idx = DualIndex(BagOfWordsEncoder(), ApplicabilityWriter(llm), index_factory=ListIndex)
    idx.add(Component(id="a", name="Alpha", description="wiring and cables"))
    idx.add(Component(id="b", name="Beta", description="billing and invoices"))

    query = "wiring and cables"
    at_zero = {m.component.id: m.score for m in idx.resolve(query, task_weight=0.0, top_k=2)}
    at_one = {m.component.id: m.score for m in idx.resolve(query, task_weight=1.0, top_k=2)}

    # "a" describes wiring; "b" is *for* wiring. The weight swaps which wins.
    assert at_zero["a"] > at_zero["b"]
    assert at_one["b"] > at_one["a"]


def test_both_indexes_are_searched_not_one(index):
    """The recall fix over the original implementation.

    A component invisible to the content index must still be reachable through
    the applicability index. Retrieving from one index and re-ranking cannot do
    this — the candidate never enters the list.
    """
    llm = ScriptedLLM({"Hidden": "exactly what the task needs"})
    idx = DualIndex(BagOfWordsEncoder(), ApplicabilityWriter(llm), index_factory=ListIndex)
    idx.add(Component(id="hidden", name="Hidden", description="zzz qqq xxx"))
    for i in range(8):
        idx.add(Component(id=f"noise{i}", name=f"Noise {i}", description="task needs words"))

    hits = idx.resolve("exactly what the task needs", task_weight=1.0, top_k=3)
    assert "hidden" in {m.component.id for m in hits}


# ── Scoring behaviour ─────────────────────────────────────────────────────────


def test_every_subscore_is_reported(index):
    index.add(Component(id="x", name="Thing", description="does things"))
    match = index.resolve("things", top_k=1)[0]

    assert 0.0 <= match.content_score <= 1.0
    assert 0.0 <= match.applicability_score <= 1.0
    assert "content" in match.explain() and "applicability" in match.explain()


def test_usage_boost_breaks_ties_but_cannot_overturn_relevance(index):
    index.add(Component(id="popular", name="Widget", description="handles widgets",
                        usage_count=1000, success_count=1000))
    index.add(Component(id="fresh", name="Widget", description="handles widgets"))

    ranked = index.resolve("widgets", top_k=2)
    assert ranked[0].component.id == "popular"
    assert ranked[0].boost <= 0.05, "a usage boost above the cap could overturn relevance"

    # And with the boost off, the tie is genuine.
    unboosted = index.resolve("widgets", top_k=2, use_boost=False)
    assert all(m.boost == 0.0 for m in unboosted)


def test_boost_needs_successes_not_just_usage(index):
    index.add(Component(id="tried", name="Widget", description="handles widgets",
                        usage_count=500, success_count=0))
    assert index.resolve("widgets", top_k=1)[0].boost == 0.0


def test_task_weight_is_validated(index):
    index.add(Component(id="x", name="X", description="x"))
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="task_weight"):
            index.resolve("x", task_weight=bad)


# ── Index maintenance ─────────────────────────────────────────────────────────


def test_empty_index_resolves_to_nothing(index):
    assert index.resolve("anything") == []


def test_readding_replaces_rather_than_duplicates(index):
    index.add(Component(id="x", name="Old name", description="old"))
    index.add(Component(id="x", name="New name", description="new"))

    assert len(index) == 1
    assert index.resolve("new", top_k=5)[0].component.name == "New name"


def test_removal_takes_it_out_of_both_indexes(index):
    index.add(Component(id="x", name="Gone", description="gone"))
    index.remove("x")

    assert len(index) == 0
    assert index.resolve("gone") == []
    index.remove("x")  # removing twice is not an error


# ── Applicability generation ──────────────────────────────────────────────────


def test_applicability_is_cached_so_indexing_does_not_re_run_the_llm():
    llm = ScriptedLLM({})
    writer = ApplicabilityWriter(llm)
    component = Component(id="x", name="X", description="does x")

    ensure_applicability(component, writer)
    ensure_applicability(component, writer)
    assert len(llm.prompts) == 1, "cached applicability should not be regenerated"

    ensure_applicability(component, writer, force=True)
    assert len(llm.prompts) == 2


def test_the_prompt_asks_for_purpose_not_description():
    llm = ScriptedLLM({})
    ApplicabilityWriter(llm).write(
        Component(id="x", name="Widget", description="handles widgets", domain="factory")
    )
    prompt = llm.prompts[0]

    assert "what this component is FOR" in prompt
    assert "Widget" in prompt and "factory" in prompt


def test_template_fallback_needs_no_llm():
    writer = ApplicabilityWriter()
    assert not writer.uses_llm

    text = writer.write(Component(id="x", name="Cable sizing", description="Parses IEC tables",
                                  domain="electrical", kind="tool"))
    assert "electrical" in text and "tool" in text
    assert text == writer.write(Component(id="x", name="Cable sizing",
                                          description="Parses IEC tables",
                                          domain="electrical", kind="tool"))


def test_code_is_included_in_the_prompt_but_truncated():
    llm = ScriptedLLM({})
    ApplicabilityWriter(llm).write(
        Component(id="x", name="X", description="d", code="A" * 900)
    )
    prompt = llm.prompts[0]
    assert "```" in prompt and "..." in prompt
    assert "A" * 600 not in prompt, "long code should be truncated before the prompt"
