"""Shared fixtures: mock LLM, in-memory SQLite, sample data factories."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from evolving_memory.config import CTEConfig
from evolving_memory.llm.base import BaseLLMProvider
from evolving_memory.models.graph import ParentNode, ChildNode, ThoughtEdge
from evolving_memory.models.hierarchy import HierarchyLevel, TraceOutcome, TraceSource, EdgeType
from evolving_memory.models.trace import ActionEntry, TraceEntry, TraceSession
from evolving_memory.storage.sqlite_store import SQLiteStore
from evolving_memory.storage.vector_index import VectorIndex


# ── Mock LLM ────────────────────────────────────────────────────────


class MockLLMProvider(BaseLLMProvider):
    """Deterministic LLM mock for testing — returns ISA opcode programs."""

    async def complete(self, prompt: str, system: str = "") -> str:
        return "Mock LLM response"

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        # Legacy method — return sensible defaults for any remaining JSON callers
        if "negative_constraints" in prompt.lower() or "failure" in prompt.lower():
            return {
                "negative_constraints": [
                    {"description": "Do not retry without backoff", "reasoning": "Causes rate limiting"}
                ]
            }
        if "critical_path" in prompt.lower() or "critical path" in prompt.lower():
            return {
                "critical_path": [
                    {"index": 0, "reasoning": "Analyze requirements", "action": "read docs", "result": "understood", "why_critical": "Foundation step"},
                    {"index": 1, "reasoning": "Implement solution", "action": "write code", "result": "working", "why_critical": "Core implementation"},
                ]
            }
        if "should_merge" in prompt.lower() or "merge" in prompt.lower():
            return {
                "should_merge": False,
                "reasoning": "Nodes are distinct",
            }
        return {}

    async def emit_program(self, prompt: str, system: str = "") -> str:
        """Return ISA opcode programs based on prompt context."""
        prompt_lower = prompt.lower()

        # Cross-trace link discovery — LNK_NODE
        if "lnk_node" in prompt_lower and "strategy a" in prompt_lower:
            return self._emit_cross_link(prompt)

        # SWS failure analysis — EXTRACT_CONSTRAINT
        if "extract_constraint" in prompt_lower and ("failure" in prompt_lower or "partial" in prompt_lower):
            # Extract trace_id from prompt
            trace_id = self._extract_trace_id(prompt)
            return (
                f'EXTRACT_CONSTRAINT {trace_id} "Do not retry without backoff"\n'
                f"HALT"
            )

        # SWS critical path — MARK_CRITICAL / MARK_NOISE
        if "mark_critical" in prompt_lower:
            trace_id = self._extract_trace_id(prompt)
            # Count actions from prompt to mark them all critical
            action_lines = [l for l in prompt.splitlines() if l.strip().startswith("[")]
            lines = []
            for i, _ in enumerate(action_lines):
                lines.append(f"MARK_CRITICAL {trace_id} {i}")
            lines.append("HALT")
            return "\n".join(lines)

        # REM — BUILD_PARENT + BUILD_CHILD
        if "build_parent" in prompt_lower:
            # Extract goal from prompt
            goal = self._extract_field(prompt, "Goal:")
            steps_text = prompt.split("Critical path steps:")[-1].split("Negative constraints:")[0] if "Critical path steps:" in prompt else ""
            step_lines = [l.strip() for l in steps_text.splitlines() if l.strip().startswith("[")]

            lines = [f'BUILD_PARENT "{goal}" "Strategy for: {goal}" 0.85']
            for i, step_line in enumerate(step_lines):
                # Parse step: [idx] reasoning → action → result
                parts = step_line.split("→")
                if len(parts) >= 3:
                    reasoning = parts[0].strip().lstrip("[0123456789] ").strip()
                    action = parts[1].strip()
                    result = parts[2].strip()
                else:
                    reasoning = "Execute step"
                    action = "perform action"
                    result = "success"
                lines.append(
                    f'BUILD_CHILD $LAST_PARENT {i} "{reasoning}" "{action}" "{result}"'
                )
            lines.append("HALT")
            return "\n".join(lines)

        # Default: just HALT
        return "HALT"

    @staticmethod
    def _emit_cross_link(prompt: str) -> str:
        """Emit LNK_NODE opcodes for cross-trace link discovery.

        Heuristic: if both strategies mention related keywords, emit a causal
        link. Otherwise emit nothing (HALT only).
        """
        id_a = ""
        id_b = ""
        goal_a = ""
        goal_b = ""
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith("Strategy A (ID:"):
                id_a = stripped.split("ID:")[1].rstrip("):").strip()
            elif stripped.startswith("Strategy B (ID:"):
                id_b = stripped.split("ID:")[1].rstrip("):").strip()
            elif stripped.startswith("Goal:"):
                if not goal_a:
                    goal_a = stripped.split(":", 1)[1].strip().lower()
                else:
                    goal_b = stripped.split(":", 1)[1].strip().lower()

        if not id_a or not id_b:
            return "HALT"

        # Simple relatedness heuristic for tests:
        # If the goals share any significant word (>3 chars), they're related
        words_a = {w for w in goal_a.split() if len(w) > 3}
        words_b = {w for w in goal_b.split() if len(w) > 3}
        if words_a & words_b:
            return f'LNK_NODE {id_a} {id_b} "causal"\nHALT'

        return "HALT"

    @staticmethod
    def _extract_trace_id(prompt: str) -> str:
        """Extract trace ID from prompt text."""
        for line in prompt.splitlines():
            if line.strip().lower().startswith("trace id:"):
                return line.split(":", 1)[1].strip()
        return "unknown_trace"

    @staticmethod
    def _extract_field(prompt: str, field: str) -> str:
        """Extract a field value from prompt text."""
        for line in prompt.splitlines():
            if line.strip().startswith(field):
                return line.split(":", 1)[1].strip()
        return "unknown"


# ── Mock Encoder ────────────────────────────────────────────────────


class MockEmbeddingEncoder:
    """Deterministic embedding encoder for testing — uses hash-based vectors."""

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2**31))
        vec = rng.randn(self._dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts])


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def mock_encoder():
    return MockEmbeddingEncoder(dim=768)


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def vector_index():
    return VectorIndex(dim=768)


@pytest.fixture
def config():
    return CTEConfig()


# ── Factories ───────────────────────────────────────────────────────


def make_trace(
    goal: str = "test goal",
    outcome: TraceOutcome = TraceOutcome.SUCCESS,
    n_actions: int = 3,
    session_id: str | None = None,
    source: TraceSource = TraceSource.UNKNOWN_SOURCE,
) -> TraceEntry:
    actions = [
        ActionEntry(
            reasoning=f"step {i} reasoning",
            action_payload=f"step {i} action",
            result=f"step {i} result",
        )
        for i in range(n_actions)
    ]
    return TraceEntry(
        hierarchy_level=HierarchyLevel.TACTICAL,
        goal=goal,
        outcome=outcome,
        confidence=0.8,
        source=source,
        action_entries=actions,
        session_id=session_id,
    )


def make_session(
    root_goal: str = "test session",
    n_traces: int = 2,
    n_actions: int = 3,
) -> TraceSession:
    session = TraceSession(root_goal=root_goal)
    for i in range(n_traces):
        trace = make_trace(
            goal=f"trace {i} goal",
            session_id=session.session_id,
            n_actions=n_actions,
        )
        session.traces.append(trace)
    session.ended_at = datetime.now(timezone.utc)
    return session


def make_parent_node(
    goal: str = "test goal",
    summary: str = "test summary",
    confidence: float = 0.8,
    outcome: TraceOutcome = TraceOutcome.SUCCESS,
) -> ParentNode:
    return ParentNode(
        hierarchy_level=HierarchyLevel.TACTICAL,
        content="test content",
        summary=summary,
        confidence=confidence,
        goal=goal,
        outcome=outcome,
        success_count=1 if outcome == TraceOutcome.SUCCESS else 0,
        failure_count=1 if outcome == TraceOutcome.FAILURE else 0,
    )


def make_child_node(parent_node_id: str, step_index: int = 0) -> ChildNode:
    return ChildNode(
        parent_node_id=parent_node_id,
        hierarchy_level=HierarchyLevel.TACTICAL,
        content=f"step {step_index} content",
        summary=f"step {step_index} summary",
        confidence=0.8,
        step_index=step_index,
        reasoning=f"step {step_index} reasoning",
        action=f"step {step_index} action",
        result=f"step {step_index} result",
    )
