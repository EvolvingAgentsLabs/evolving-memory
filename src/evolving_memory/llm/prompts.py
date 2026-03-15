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

# ── Phase 3b: Cross-Trace Link Discovery ──────────────────────────

CONSOLIDATION_LINK_SYSTEM = """\
You are a knowledge graph architect for a memory consolidation system.
Output ONLY LNK_NODE instructions, one per line. No prose, no explanation.
Lines starting with # are comments (optional)."""

CONSOLIDATION_LINK_DISCOVERY = """\
Analyze these two memory strategies and determine if they are conceptually connected.

Strategy A (ID: {id_a}):
  Goal: {goal_a}
  Summary: {summary_a}
  Steps: {steps_a}

Strategy B (ID: {id_b}):
  Goal: {goal_b}
  Summary: {summary_b}
  Steps: {steps_b}

If A is a prerequisite for B (learning A helps understand B):
  LNK_NODE {id_a} {id_b} "causal"

If B is a prerequisite for A:
  LNK_NODE {id_b} {id_a} "causal"

If they share concepts but are different approaches/topics:
  LNK_NODE {id_a} {id_b} "context_jump"

If they are unrelated, emit nothing.
End with HALT."""
