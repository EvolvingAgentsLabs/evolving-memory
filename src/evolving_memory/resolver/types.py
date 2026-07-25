"""Types for task-aware component resolution."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Component:
    """Something the resolver can retrieve: a skill, tool, agent, or snippet.

    Two texts describe it, and the distinction is the whole point of this module:

    * ``content`` — what the thing *is*. Its own words: name, description, code.
    * ``applicability`` — what the thing is *for*. Which tasks it solves, who
      would reach for it, when it is the right choice.

    Retrieval systems that index only the first answer "what looks like this
    query". Indexing both lets you ask "what would help me do this", which is
    the question a task actually poses.
    """

    id: str
    name: str
    description: str = ""
    kind: str = "component"
    domain: str = "general"
    code: str = ""

    #: Filled by an :class:`ApplicabilityWriter`. Cached so re-indexing does not
    #: re-run an LLM over components that have not changed.
    applicability: str = ""

    #: Feedback signal, used for a small ranking boost. Kept deliberately small:
    #: popularity should break ties, never overturn relevance.
    usage_count: int = 0
    success_count: int = 0

    metadata: dict = field(default_factory=dict)

    def content_text(self) -> str:
        """The text describing what this component is."""
        parts = [self.name, self.description]
        if self.code:
            parts.append(self.code[:2000])
        return "\n".join(p for p in parts if p)

    @property
    def success_rate(self) -> float:
        if self.usage_count <= 0:
            return 0.0
        return self.success_count / self.usage_count


@dataclass
class Match:
    """One resolved component and why it ranked where it did.

    Every sub-score is kept rather than collapsed into the final number, because
    the interesting failures are the ones where a component wins on content and
    loses on applicability — and you cannot see that from a single float.
    """

    component: Component
    score: float
    content_score: float
    applicability_score: float
    boost: float = 0.0

    def explain(self) -> str:
        return (
            f"{self.component.name}: {self.score:.3f} "
            f"(content {self.content_score:.3f}, applicability "
            f"{self.applicability_score:.3f}, boost {self.boost:+.3f})"
        )
