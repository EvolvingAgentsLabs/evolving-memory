"""VMContext — execution state for the Cognitive VM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VMContext:
    """Mutable execution state carried through a VM program run.

    The VM accumulates results here; the dream engine persists them after
    the VM completes.  No database commits happen mid-program.
    """

    # References to infrastructure (set by CognitiveVM before execution)
    store: Any = None
    index: Any = None
    encoder: Any = None

    # Accumulator — result of the last executed instruction
    accumulator: Any = None

    # Output buffer — messages from YIELD instructions
    output: list[str] = field(default_factory=list)

    # Working memory — populated by dream/consolidation handlers
    critical_indices: list[tuple[str, int]] = field(default_factory=list)  # (trace_id, action_index)
    noise_indices: list[tuple[str, int]] = field(default_factory=list)     # (trace_id, action_index)
    constraints: list[tuple[str, str]] = field(default_factory=list)       # (trace_id, description)
    built_parents: list[dict[str, Any]] = field(default_factory=list)      # parent node data
    built_children: list[dict[str, Any]] = field(default_factory=list)     # child node data
    built_edges: list[dict[str, str]] = field(default_factory=list)        # edge data

    # Safety
    max_instructions: int = 500
    instructions_executed: int = 0

    # Audit log
    side_effects: list[str] = field(default_factory=list)


@dataclass
class VMResult:
    """Result of a complete VM program execution."""
    success: bool = True
    instructions_executed: int = 0
    output: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    error: str | None = None

    # Collected working memory (copied from context at end)
    critical_indices: list[tuple[str, int]] = field(default_factory=list)
    noise_indices: list[tuple[str, int]] = field(default_factory=list)
    constraints: list[tuple[str, str]] = field(default_factory=list)
    built_parents: list[dict] = field(default_factory=list)
    built_children: list[dict] = field(default_factory=list)
    built_edges: list[dict[str, str]] = field(default_factory=list)
    accumulator: Any = None
