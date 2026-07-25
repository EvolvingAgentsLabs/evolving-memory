"""Task-aware component resolution over two embeddings.

    from evolving_memory.resolver import Component, DualIndex

    index = DualIndex(encoder)
    index.add(Component(id="cable", name="Cable sizing tables",
                        description="Parses IEC 60364 conductor tables"))

    index.resolve("check if this wiring is up to code", task_weight=0.7)

Recovered from the Evolving Agents Toolkit's SmartLibrary, minus MongoDB.
See ``dual_index`` for what changed on the way back.
"""

from .applicability import PROMPT, ApplicabilityWriter, ensure_applicability
from .dual_index import MAX_BOOST, DualIndex
from .types import Component, Match

__all__ = [
    "Component",
    "Match",
    "DualIndex",
    "ApplicabilityWriter",
    "ensure_applicability",
    "PROMPT",
    "MAX_BOOST",
]
