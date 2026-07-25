"""Producing the *what it is for* text that the second index embeds.

The naive version of task-aware retrieval embeds a component's ``description``
field and calls it done. That fails in a specific, reproducible way: a
description says what a thing is, and a task says what someone needs. "Parses
IEC 60364 cable tables" and "figure out if this wiring is up to code" share
almost no surface, and a description index scores them apart.

So the applicability text is written *for the task side of that gap*: use cases,
who reaches for this, when it is the right choice, what it pairs with.
"""

from __future__ import annotations

from .types import Component

PROMPT = """\
Analyze this component.

Name: {name}
Type: {kind}
Domain: {domain}
Description: {description}
{code_block}
Write a short paragraph describing what this component is FOR — not what it is.
Cover, in prose and without headings:

1. The specific problems or tasks it is designed to solve.
2. Who or what would reach for it — a developer, a pipeline, another agent.
3. Its one or two real strengths for that purpose.
4. The situations where it is the right choice, and where it is not.
5. What it typically works alongside.

Write it the way someone would describe the component when recommending it for a
job, using the vocabulary of the job rather than of the implementation. Output
only the paragraph.
"""


class ApplicabilityWriter:
    """Generates the applicability text for a component.

    Pass an ``llm`` with a ``generate(prompt) -> str`` method for the real thing.
    Without one it falls back to a deterministic template, which keeps indexing
    runnable and tests hermetic — an LLM call per component at index time is
    both expensive and non-reproducible, and a resolver you cannot test offline
    is a resolver nobody will benchmark.
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm

    @property
    def uses_llm(self) -> bool:
        return self._llm is not None

    def write(self, component: Component) -> str:
        if self._llm is None:
            return self._template(component)

        code_block = ""
        if component.code:
            snippet = component.code[:500]
            if len(component.code) > 500:
                snippet += "..."
            code_block = f"Code summary:\n```\n{snippet}\n```\n"

        prompt = PROMPT.format(
            name=component.name,
            kind=component.kind,
            domain=component.domain,
            description=component.description or "(none given)",
            code_block=code_block,
        )
        return str(self._llm.generate(prompt)).strip()

    @staticmethod
    def _template(component: Component) -> str:
        """A deterministic stand-in.

        Weaker than a written applicability text — it cannot invent the
        task-side vocabulary that makes the second index earn its place. It
        exists so the pipeline runs without a key, and so tests measure the
        retrieval maths rather than a model's mood on the day.
        """
        bits = [
            f"Use this when you need to {component.description.rstrip('.').lower()}."
            if component.description
            else f"Use this for {component.name} tasks.",
            f"It applies in the {component.domain} domain.",
            f"It is a {component.kind}, reached for when that capability is the "
            f"missing piece of a task rather than the goal itself.",
        ]
        return " ".join(bits)


def ensure_applicability(
    component: Component, writer: ApplicabilityWriter, *, force: bool = False
) -> Component:
    """Fill ``component.applicability`` if it is empty. Returns the component."""
    if force or not component.applicability:
        component.applicability = writer.write(component)
    return component
