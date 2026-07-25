"""Task-aware retrieval over two embeddings per component.

Recovered from the Evolving Agents Toolkit, which indexed every component twice
— once for what it is, once for what it is for — and resolved tasks against the
second. The idea was right and the implementation needed MongoDB Atlas Vector
Search, so it did not survive the move to a local stack. The ecosystem that
replaced it resolves skills by matching a description field, which is the naive
form of the same idea.

Two things changed on the way back.

**No infrastructure.** Two in-process vector indexes and any encoder exposing
``encode(text) -> vector`` and ``dim``. Nothing to provision.

**Both indexes are searched, not one.** EAT picked an index based on whether a
task context was supplied, retrieved from it, then re-scored the survivors using
the other embedding. That cannot rank a component the first index missed — and
the components most worth finding are exactly the ones whose *description* looks
nothing like the task. Searching both and scoring the union fixes a recall hole
that re-ranking alone cannot reach.
"""

from __future__ import annotations

import numpy as np

from .applicability import ApplicabilityWriter, ensure_applicability
from .types import Component, Match

#: How much a component's usage record may move it. Deliberately small: a
#: popular component should break a tie, never overturn relevance.
MAX_BOOST = 0.05


class DualIndex:
    """Two vector indexes over one set of components.

    :param encoder: anything with ``encode(text) -> np.ndarray`` and ``dim``.
    :param writer: produces applicability text. Defaults to the template writer,
        which needs no LLM.
    :param index_factory: builds a vector index. Defaults to
        :class:`~evolving_memory.storage.vector_index.VectorIndex`, injectable so
        tests do not need faiss.
    """

    def __init__(self, encoder, writer: ApplicabilityWriter | None = None, index_factory=None):
        self._encoder = encoder
        self._writer = writer or ApplicabilityWriter()

        if index_factory is None:
            from ..storage.vector_index import VectorIndex

            index_factory = lambda dim: VectorIndex(dim=dim)  # noqa: E731

        self._content = index_factory(encoder.dim)
        self._applicability = index_factory(encoder.dim)
        self._components: dict[str, Component] = {}
        self._vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self._components)

    @property
    def components(self) -> list[Component]:
        return list(self._components.values())

    def add(self, component: Component, *, force_applicability: bool = False) -> Component:
        """Index one component under both embeddings."""
        ensure_applicability(component, self._writer, force=force_applicability)

        content_vec = np.asarray(self._encoder.encode(component.content_text()), dtype=np.float32)
        applic_vec = np.asarray(self._encoder.encode(component.applicability), dtype=np.float32)

        if component.id in self._components:
            self._content.remove(component.id)
            self._applicability.remove(component.id)

        self._content.add(component.id, content_vec)
        self._applicability.add(component.id, applic_vec)
        self._components[component.id] = component
        self._vectors[component.id] = (content_vec, applic_vec)
        return component

    def add_all(self, components, **kwargs) -> list[Component]:
        return [self.add(c, **kwargs) for c in components]

    def remove(self, component_id: str) -> None:
        if component_id not in self._components:
            return
        self._content.remove(component_id)
        self._applicability.remove(component_id)
        del self._components[component_id]
        del self._vectors[component_id]

    def resolve(
        self,
        query: str,
        *,
        task: str | None = None,
        task_weight: float = 0.5,
        top_k: int = 5,
        candidate_multiplier: int = 4,
        use_boost: bool = True,
    ) -> list[Match]:
        """Rank components against a query, and optionally against a task.

        :param query: what is being looked for.
        :param task: the job it is needed for. When omitted, ``query`` plays both
            parts — a component whose applicability matches the query still
            scores on that axis.
        :param task_weight: 0 ranks purely on what components *are*, 1 purely on
            what they are *for*. The default weighs them evenly.
        :param candidate_multiplier: how deep to pull from each index before
            scoring. Retrieval is approximate; scoring is not.
        """
        if not self._components:
            return []
        if not 0.0 <= task_weight <= 1.0:
            raise ValueError(f"task_weight must be in [0, 1], got {task_weight}")

        query_vec = np.asarray(self._encoder.encode(query), dtype=np.float32)
        task_vec = np.asarray(self._encoder.encode(task), dtype=np.float32) if task else query_vec

        # Union of both indexes. A component that is a poor lexical match for
        # the query but an excellent fit for the task lives only in the second
        # list, and that component is the entire reason for this module.
        depth = max(top_k * candidate_multiplier, top_k)
        candidates: set[str] = set()
        for index, vec in ((self._content, query_vec), (self._applicability, task_vec)):
            for cid, _ in index.search(vec, top_k=depth):
                if cid in self._components:
                    candidates.add(cid)

        matches = []
        for cid in candidates:
            component = self._components[cid]
            content_vec, applic_vec = self._vectors[cid]

            content_score = _cosine(query_vec, content_vec)
            applicability_score = _cosine(task_vec, applic_vec)
            score = task_weight * applicability_score + (1.0 - task_weight) * content_score

            boost = 0.0
            if use_boost and component.usage_count > 0:
                boost = min(MAX_BOOST, (component.usage_count / 200.0) * component.success_rate)

            matches.append(
                Match(
                    component=component,
                    score=min(1.0, score + boost),
                    content_score=content_score,
                    applicability_score=applicability_score,
                    boost=boost,
                )
            )

        matches.sort(key=lambda m: (-m.score, m.component.id))
        return matches[:top_k]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. The encoder normalises, but callers may not."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
