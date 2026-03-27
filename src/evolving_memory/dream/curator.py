"""Phase 1 — SWS (Slow-Wave Sleep): failure analysis and critical path extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..isa.parser import InstructionParser
from ..llm.base import BaseLLMProvider
from ..llm.prompts import SWS_SYSTEM, SWS_FAILURE_ANALYSIS, SWS_CRITICAL_PATH
from ..models.hierarchy import TraceOutcome
from ..models.trace import TraceEntry
from ..vm.machine import CognitiveVM

logger = logging.getLogger(__name__)


@dataclass
class CriticalStep:
    index: int
    reasoning: str
    action: str
    result: str


@dataclass
class CuratedTrace:
    """Output of Phase 1 — a trace reduced to its critical path + constraints."""
    trace: TraceEntry
    critical_steps: list[CriticalStep] = field(default_factory=list)
    negative_constraints: list[tuple[str, str]] = field(default_factory=list)  # (description, failure_class)


class TraceCurator:
    """Phase 1 SWS — curates raw traces by extracting failures and critical paths."""

    def __init__(self, llm: BaseLLMProvider) -> None:
        self._llm = llm
        self._parser = InstructionParser()

    async def curate(self, traces: list[TraceEntry], min_actions: int = 2) -> list[CuratedTrace]:
        results = []
        for trace in traces:
            if len(trace.action_entries) < min_actions:
                continue
            curated = await self._curate_single(trace)
            results.append(curated)
        return results

    async def _curate_single(self, trace: TraceEntry) -> CuratedTrace:
        actions_text = self._format_actions(trace)
        constraints: list[str] = []

        # Extract negative constraints from failures via ISA
        if trace.outcome in (TraceOutcome.FAILURE, TraceOutcome.PARTIAL):
            try:
                raw = await self._llm.emit_program(
                    SWS_FAILURE_ANALYSIS.format(
                        trace_id=trace.trace_id,
                        goal=trace.goal,
                        outcome=trace.outcome.value,
                        actions=actions_text,
                    ),
                    system=SWS_SYSTEM,
                )
                program = self._parser.parse(raw)
                vm = CognitiveVM()
                result = vm.execute(program)
                constraints = [(desc, fc) for _, desc, fc in result.constraints]
            except Exception:
                logger.debug("SWS failure analysis failed for %s", trace.trace_id)

        # Extract critical path via ISA
        critical_steps: list[CriticalStep] = []
        try:
            raw = await self._llm.emit_program(
                SWS_CRITICAL_PATH.format(
                    trace_id=trace.trace_id,
                    goal=trace.goal,
                    outcome=trace.outcome.value,
                    actions=actions_text,
                ),
                system=SWS_SYSTEM,
            )
            program = self._parser.parse(raw)
            vm = CognitiveVM()
            result = vm.execute(program)

            # Collect noise indices for filtering
            noise_set = {idx for _, idx in result.noise_indices}

            # Build CriticalStep from MARK_CRITICAL results, excluding noise
            for trace_id, action_index in result.critical_indices:
                if action_index in noise_set:
                    continue  # Intelligent forgetting: skip noisy actions
                if action_index < len(trace.action_entries):
                    a = trace.action_entries[action_index]
                    critical_steps.append(CriticalStep(
                        index=action_index,
                        reasoning=a.reasoning,
                        action=a.action_payload,
                        result=a.result,
                    ))
        except Exception:
            logger.debug("SWS critical path failed for %s, using fallback", trace.trace_id)

        # Fallback: treat all actions as critical
        if not critical_steps:
            for i, action in enumerate(trace.action_entries):
                critical_steps.append(CriticalStep(
                    index=i,
                    reasoning=action.reasoning,
                    action=action.action_payload,
                    result=action.result,
                ))

        return CuratedTrace(
            trace=trace,
            critical_steps=critical_steps,
            negative_constraints=constraints,
        )

    @staticmethod
    def _format_actions(trace: TraceEntry) -> str:
        lines = []
        for i, a in enumerate(trace.action_entries):
            lines.append(f"[{i}] Reasoning: {a.reasoning}")
            lines.append(f"    Action: {a.action_payload}")
            lines.append(f"    Result: {a.result}")
        return "\n".join(lines)
