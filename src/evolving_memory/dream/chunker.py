"""Phase 2 — REM: hierarchical parent+child node creation from curated traces."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..isa.parser import InstructionParser
from ..llm.base import BaseLLMProvider
from ..llm.prompts import REM_SYSTEM, REM_BUILD_NODES
from ..models.graph import ParentNode, ChildNode
from ..models.hierarchy import HierarchyLevel, TraceOutcome
from ..vm.machine import CognitiveVM
from .curator import CuratedTrace

logger = logging.getLogger(__name__)


@dataclass
class ChunkedResult:
    """Output of Phase 2 — a parent node with its child nodes."""
    parent: ParentNode
    children: list[ChildNode] = field(default_factory=list)
    source_trace_id: str = ""


class HierarchicalChunker:
    """Phase 2 REM — creates hierarchical parent+child nodes from curated traces.

    Key improvement over JSON approach: emits BUILD_PARENT + BUILD_CHILD in a
    single LLM call (1 call instead of 1+N).
    """

    def __init__(self, llm: BaseLLMProvider) -> None:
        self._llm = llm
        self._parser = InstructionParser()

    async def chunk(self, curated_traces: list[CuratedTrace]) -> list[ChunkedResult]:
        results = []
        for curated in curated_traces:
            if not curated.critical_steps:
                continue
            result = await self._chunk_single(curated)
            results.append(result)
        return results

    async def _chunk_single(self, curated: CuratedTrace) -> ChunkedResult:
        trace = curated.trace
        steps_text = self._format_steps(curated)

        # Single LLM call: emit BUILD_PARENT + BUILD_CHILD opcodes
        try:
            raw = await self._llm.emit_program(
                REM_BUILD_NODES.format(
                    goal=trace.goal,
                    outcome=trace.outcome.value,
                    steps=steps_text,
                    constraints=", ".join(curated.negative_constraints) or "none",
                ),
                system=REM_SYSTEM,
            )
            program = self._parser.parse(raw)
            vm = CognitiveVM()
            result = vm.execute(program)
        except Exception:
            logger.debug("REM chunking failed for %s, using fallback", trace.trace_id)
            result = None

        # Build parent from VM result or fallback
        if result and result.built_parents:
            pd = result.built_parents[0]
            parent = ParentNode(
                hierarchy_level=trace.hierarchy_level,
                content=steps_text,
                summary=pd.get("summary", trace.goal),
                confidence=pd.get("confidence", trace.confidence),
                goal=trace.goal,
                outcome=trace.outcome,
                trigger_goals=[trace.goal],
                negative_constraints=curated.negative_constraints,
                success_count=1 if trace.outcome == TraceOutcome.SUCCESS else 0,
                failure_count=1 if trace.outcome == TraceOutcome.FAILURE else 0,
            )
        else:
            parent = ParentNode(
                hierarchy_level=trace.hierarchy_level,
                content=steps_text,
                summary=f"Strategy for: {trace.goal}",
                confidence=trace.confidence,
                goal=trace.goal,
                outcome=trace.outcome,
                trigger_goals=[trace.goal],
                negative_constraints=curated.negative_constraints,
                success_count=1 if trace.outcome == TraceOutcome.SUCCESS else 0,
                failure_count=1 if trace.outcome == TraceOutcome.FAILURE else 0,
            )

        # Build children from VM result or fallback
        children: list[ChildNode] = []
        if result and result.built_children:
            for cd in result.built_children:
                child = ChildNode(
                    parent_node_id=parent.node_id,
                    hierarchy_level=parent.hierarchy_level,
                    content=f"{cd['reasoning']} → {cd['action']} → {cd['result']}",
                    summary=f"Step {cd['step_index']}: {cd['action']}",
                    confidence=parent.confidence,
                    step_index=cd["step_index"],
                    reasoning=cd["reasoning"],
                    action=cd["action"],
                    result=cd["result"],
                    is_critical_path=True,
                )
                children.append(child)
        else:
            # Fallback: create children from critical steps directly
            for i, step in enumerate(curated.critical_steps):
                child = ChildNode(
                    parent_node_id=parent.node_id,
                    hierarchy_level=parent.hierarchy_level,
                    content=f"{step.reasoning} → {step.action} → {step.result}",
                    summary=f"Step {i + 1}: {step.action}",
                    confidence=parent.confidence,
                    step_index=i,
                    reasoning=step.reasoning,
                    action=step.action,
                    result=step.result,
                    is_critical_path=True,
                )
                children.append(child)

        parent.child_node_ids = [c.node_id for c in children]

        return ChunkedResult(
            parent=parent,
            children=children,
            source_trace_id=trace.trace_id,
        )

    @staticmethod
    def _format_steps(curated: CuratedTrace) -> str:
        lines = []
        for step in curated.critical_steps:
            lines.append(f"[{step.index}] {step.reasoning} → {step.action} → {step.result}")
        return "\n".join(lines)
