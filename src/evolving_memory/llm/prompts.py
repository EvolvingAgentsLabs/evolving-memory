"""Prompt templates for the dream engine's LLM calls — ISA opcode format."""

# ── Phase 1: SWS (TraceCurator) ────────────────────────────────────

SWS_SYSTEM = """\
You are a cognitive instruction emitter for a memory consolidation system.
Output ONLY instructions, one per line. No prose, no explanation.
Lines starting with # are comments (optional)."""

SWS_FAILURE_ANALYSIS = """\
Analyze this execution trace that ended in {outcome}.
Extract negative constraints — things the agent should NOT do in similar situations.

Trace ID: {trace_id}
Goal: {goal}
Actions:
{actions}

Available instruction:
  EXTRACT_CONSTRAINT <trace_id> "<description>"

Emit one EXTRACT_CONSTRAINT per anti-pattern found. End with HALT."""

SWS_CRITICAL_PATH = """\
Given this execution trace, identify the CRITICAL PATH — the minimal
sequence of actions essential to achieving the goal.

Trace ID: {trace_id}
Goal: {goal}
Outcome: {outcome}
Actions:
{actions}

Available instructions:
  MARK_CRITICAL <trace_id> <action_index>
  MARK_NOISE <trace_id> <action_index>

Emit MARK_CRITICAL for each essential action index.
Emit MARK_NOISE for each non-essential action index.
End with HALT."""

# ── Phase 2: REM (HierarchicalChunker) ─────────────────────────────

REM_SYSTEM = """\
You are a cognitive instruction emitter for a memory consolidation system.
Output ONLY instructions, one per line. No prose, no explanation.
Lines starting with # are comments (optional)."""

REM_BUILD_NODES = """\
Create a hierarchical memory structure from this curated trace.

Goal: {goal}
Outcome: {outcome}
Critical path steps:
{steps}
Negative constraints: {constraints}

Available instructions:
  BUILD_PARENT "<goal>" "<summary>" <confidence>
  BUILD_CHILD $LAST_PARENT <step_index> "<reasoning>" "<action>" "<result>"

First emit exactly one BUILD_PARENT with a concise summary and confidence (0.0-1.0).
Then emit one BUILD_CHILD per critical step, using $LAST_PARENT as the parent reference.
End with HALT."""

# ── Phase 3: Consolidation (TopologicalConnector) ──────────────────
# NOTE: Consolidation (connector.py) remains 100% algorithmic.
# The prompts below are kept for potential future use but are not
# currently called by the dream engine.

CONSOLIDATION_SYSTEM = """\
You are a memory consolidation engine. Your job is to determine whether two
memory nodes represent the same or overlapping knowledge."""

CONSOLIDATION_MERGE_CHECK = """\
Do these two memory nodes represent the same strategy/knowledge and should be merged?

Node A:
  Goal: {goal_a}
  Summary: {summary_a}
  Confidence: {confidence_a}

Node B:
  Goal: {goal_b}
  Summary: {summary_b}
  Confidence: {confidence_b}

Respond with JSON:
{{
  "should_merge": true/false,
  "reasoning": "...",
  "merged_summary": "..." (only if should_merge is true)
}}"""
