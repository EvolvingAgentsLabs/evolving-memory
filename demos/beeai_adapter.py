"""BeeAI ↔ Evolving Memory adapter.

Bridges BeeAI's ReActAgent event system with evolving-memory's TraceLogger,
enabling automatic trace capture and memory-guided system prompt injection.

Usage:
    from beeai_adapter import EvolvingMemoryAdapter

    adapter = EvolvingMemoryAdapter(cte)

    # Get memory context to inject into agent
    enhancement = adapter.get_memory_context("analyze Q3 sales")

    # Run agent with trace capture
    metrics = await adapter.run_with_tracing(agent, prompt, session_goal="sales analysis")
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel

from evolving_memory import (
    CognitiveTrajectoryEngine,
    HierarchyLevel,
    RouterPath,
)
from evolving_memory.models.hierarchy import TraceOutcome


# ── BeeAI ChatModel for Gemini via LiteLLM ──────────────────────────

class GeminiChatModel(LiteLLMChatModel):
    """BeeAI ChatModel using Google Gemini via LiteLLM's native gemini/ provider.

    Requires GEMINI_API_KEY environment variable.
    """

    @property
    def provider_id(self):
        return "gemini"

    def __init__(self, model_id: str = "gemini-2.5-flash-lite", settings: dict | None = None):
        # Stop at "Function Output:" to prevent model from generating past tool call boundary
        _settings = {"stop": ["Function Output:"]}
        if settings:
            _settings.update(settings)
        super().__init__(model_id, provider_id="gemini", settings=_settings)


def _safe_get(obj, key: str):
    """Extract a field from an object that may be a dict or have attributes."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


@dataclass
class RunMetrics:
    """Metrics collected from a single agent run."""
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    success: bool = False
    final_answer: str = ""
    iterations_log: list[dict] = field(default_factory=list)


class EvolvingMemoryAdapter:
    """Bridges BeeAI agent execution with evolving-memory trace capture."""

    def __init__(self, cte: CognitiveTrajectoryEngine) -> None:
        self.cte = cte

    def get_memory_context(self, prompt: str) -> tuple[str, str]:
        """Query CTE and return (system_prompt_enhancement, route_path).

        Returns:
            Tuple of (enhancement_text, route_path_string).
            Enhancement is empty string if ZERO_SHOT.
        """
        decision = self.cte.query(prompt)

        if decision.path != RouterPath.MEMORY_TRAVERSAL or not decision.entry_point:
            return "", decision.path.value

        ep = decision.entry_point
        parent = ep.parent_node

        # Build strategy from traversal
        state = self.cte.begin_traversal(ep)
        steps = []
        while True:
            child, state = self.cte.next_step(state)
            if child is None:
                break
            steps.append(child)

        # Build system prompt enhancement
        lines = [
            "IMPORTANT — I have prior experience with this type of task.",
            f"Strategy: {parent.summary}",
            "",
            "Step-by-step procedure that worked before:",
        ]
        for s in steps:
            lines.append(f"  {s.step_index + 1}. {s.reasoning} → {s.action}")
            if s.result:
                lines.append(f"     Result: {s.result[:200]}")

        if parent.negative_constraints:
            lines.append("")
            lines.append("CRITICAL CONSTRAINTS (avoid these mistakes):")
            for c in parent.negative_constraints:
                lines.append(f"  - DO NOT: {c}")

        return "\n".join(lines), decision.path.value

    async def run_with_tracing(
        self,
        agent,
        prompt: str,
        session_goal: str,
        memory_enhancement: str = "",
        max_retries_per_step: int = 3,
        max_iterations: int = 20,
        total_max_retries: int = 10,
    ) -> RunMetrics:
        """Run a BeeAI agent and capture the execution as an evolving-memory trace.

        Args:
            agent: A BeeAI ReActAgent instance.
            prompt: The user prompt to send.
            session_goal: Root goal for the trace session.
            memory_enhancement: Optional system prompt enhancement from get_memory_context().
            max_retries_per_step: BeeAI retry config.
            max_iterations: BeeAI iteration limit.
            total_max_retries: BeeAI total retry limit.

        Returns:
            RunMetrics with step counts, token usage, latency, and success status.
        """
        metrics = RunMetrics()

        # Build the effective prompt
        effective_prompt = prompt
        if memory_enhancement:
            effective_prompt = (
                f"[MEMORY CONTEXT]\n{memory_enhancement}\n\n"
                f"[TASK]\n{prompt}"
            )

        # Run agent
        start_time = time.monotonic()
        try:
            output = await agent.run(
                effective_prompt,
                execution={"max_retries_per_step": max_retries_per_step,
                            "max_iterations": max_iterations,
                            "total_max_retries": total_max_retries},
            )
            metrics.success = True

            # Extract final answer
            if hasattr(output, "result") and output.result:
                final_msg = output.result
                if hasattr(final_msg, "text"):
                    metrics.final_answer = final_msg.text
                else:
                    metrics.final_answer = str(final_msg)

            # Extract iteration data from output.iterations (reliable post-run)
            if hasattr(output, "iterations"):
                for iteration in output.iterations:
                    state = iteration.state if hasattr(iteration, "state") else iteration
                    entry = {}

                    # Extract fields — state can be an object with attributes or a dict
                    thought = _safe_get(state, "thought")
                    tool_name = _safe_get(state, "tool_name")
                    tool_input = _safe_get(state, "tool_input")
                    tool_output = _safe_get(state, "tool_output")
                    final_answer = _safe_get(state, "final_answer")

                    if thought:
                        entry["thought"] = str(thought)[:500]
                    if tool_name:
                        entry["tool_name"] = str(tool_name)
                        metrics.tool_calls += 1
                    if tool_input:
                        entry["tool_input"] = str(tool_input)[:500]
                    if tool_output:
                        output_str = str(tool_output)
                        entry["tool_output"] = output_str[:500]
                        if any(err in output_str.lower() for err in [
                            "error", "traceback", "exception", "importerror",
                        ]):
                            metrics.tool_errors += 1
                    if final_answer:
                        entry["final_answer"] = str(final_answer)[:500]

                    if entry:
                        metrics.iterations_log.append(entry)

                metrics.steps = len(metrics.iterations_log)

            # Extract token usage
            if hasattr(output, "usage") and output.usage:
                usage = output.usage
                if hasattr(usage, "prompt_tokens"):
                    metrics.prompt_tokens = usage.prompt_tokens or 0
                if hasattr(usage, "completion_tokens"):
                    metrics.completion_tokens = usage.completion_tokens or 0
                metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens

        except Exception as e:
            metrics.success = False
            metrics.final_answer = f"AGENT FAILED: {e}"

        metrics.latency_s = time.monotonic() - start_time

        # Capture trace in evolving-memory
        self._capture_trace(session_goal, prompt, metrics)

        # Estimate tokens if not reported by the backend
        if metrics.total_tokens == 0:
            metrics.total_tokens = self._estimate_tokens(prompt, metrics)

        return metrics

    def _capture_trace(
        self, session_goal: str, prompt: str, metrics: RunMetrics
    ) -> None:
        """Record the agent run as an evolving-memory trace."""
        with self.cte.session(session_goal) as logger:
            with logger.trace(HierarchyLevel.TACTICAL, prompt) as ctx:
                for entry in metrics.iterations_log:
                    reasoning = entry.get("thought", "Agent reasoning")
                    tool_name = entry.get("tool_name", "")
                    tool_input = entry.get("tool_input", "")
                    tool_output = entry.get("tool_output", "")
                    final = entry.get("final_answer", "")

                    if tool_name:
                        action_payload = f"{tool_name}({tool_input})"
                        result = tool_output
                    elif final:
                        action_payload = "final_answer"
                        result = final
                    else:
                        action_payload = "thinking"
                        result = ""

                    ctx.action(
                        reasoning=reasoning,
                        action_payload=action_payload,
                        result=result,
                    )

                if metrics.success:
                    ctx.set_outcome(TraceOutcome.SUCCESS, confidence=0.9)
                elif metrics.steps >= 15:
                    ctx.set_outcome(TraceOutcome.ABORTED, confidence=1.0)
                else:
                    ctx.set_outcome(TraceOutcome.FAILURE, confidence=1.0)

    @staticmethod
    def _estimate_tokens(prompt: str, metrics: RunMetrics) -> int:
        """Rough token estimate based on iteration content (~4 chars per token).

        Only counts iteration tokens (thoughts, tool I/O, answers) to avoid
        inflating counts when memory context is injected into the prompt.
        """
        # Estimate per-iteration token cost (prompt + response per step)
        per_step_prompt = len(prompt) // 4  # base prompt repeated each step
        iter_tokens = sum(
            len(str(e.get("thought", "")))
            + len(str(e.get("tool_input", "")))
            + len(str(e.get("tool_output", "")))
            + len(str(e.get("final_answer", "")))
            for e in metrics.iterations_log
        ) // 4
        return per_step_prompt + iter_tokens


def _estimate_cost_usd(tokens: int, price_per_million: float = 0.15) -> float:
    """Estimate API cost in USD given token count and price per 1M tokens."""
    return tokens * price_per_million / 1_000_000


def print_metrics_comparison(
    label_a: str,
    metrics_a: RunMetrics,
    label_b: str,
    metrics_b: RunMetrics,
) -> None:
    """Print a formatted comparison table of two runs with cost projections."""
    def improvement(a: float, b: float) -> str:
        if a == 0:
            return "N/A"
        pct = ((b - a) / a) * 100
        if pct < 0:
            return f"{pct:.0f}%"
        return f"+{pct:.0f}%"

    cost_a = _estimate_cost_usd(metrics_a.total_tokens)
    cost_b = _estimate_cost_usd(metrics_b.total_tokens)

    rows = [
        ("ReAct Steps", metrics_a.steps, metrics_b.steps),
        ("Tool Calls", metrics_a.tool_calls, metrics_b.tool_calls),
        ("Tool Errors", metrics_a.tool_errors, metrics_b.tool_errors),
        ("Total Tokens (est)", metrics_a.total_tokens, metrics_b.total_tokens),
        ("Est. Cost / task", f"${cost_a:.4f}", f"${cost_b:.4f}"),
        ("Latency (seconds)", f"{metrics_a.latency_s:.1f}", f"{metrics_b.latency_s:.1f}"),
        ("Success", "Yes" if metrics_a.success else "No", "Yes" if metrics_b.success else "No"),
    ]

    col_w = [28, 20, 20, 14]
    header = f"| {'Metric':<{col_w[0]}} | {label_a:>{col_w[1]}} | {label_b:>{col_w[1]}} | {'Change':>{col_w[3]}} |"
    sep = f"|{'-' * (col_w[0] + 2)}|{'-' * (col_w[1] + 2)}|{'-' * (col_w[1] + 2)}|{'-' * (col_w[3] + 2)}|"

    print()
    print(sep)
    print(header)
    print(sep)

    for label, val_a, val_b in rows:
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            change = improvement(float(val_a), float(val_b))
            print(f"| {label:<{col_w[0]}} | {str(val_a):>{col_w[1]}} | {str(val_b):>{col_w[1]}} | {change:>{col_w[3]}} |")
        else:
            print(f"| {label:<{col_w[0]}} | {str(val_a):>{col_w[1]}} | {str(val_b):>{col_w[1]}} | {'':>{col_w[3]}} |")

    print(sep)

    # Enterprise cost projection
    if metrics_a.total_tokens > 0 and metrics_b.total_tokens < metrics_a.total_tokens:
        monthly_tasks = 100_000
        monthly_save = (cost_a - cost_b) * monthly_tasks
        yearly_save = monthly_save * 12
        print()
        print(f"  ENTERPRISE COST PROJECTION (at {monthly_tasks:,} tasks/month):")
        print(f"    Without Memory:  ${cost_a * monthly_tasks:,.2f}/month")
        print(f"    With Memory:     ${cost_b * monthly_tasks:,.2f}/month")
        print(f"    Monthly Savings: ${monthly_save:,.2f}/month")
        print(f"    Annual Savings:  ${yearly_save:,.2f}/year")

    print()
