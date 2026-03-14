"""Handler functions — one per opcode — for the Cognitive VM."""

from __future__ import annotations

import logging
from typing import Any

from ..isa.opcodes import Instruction, Opcode
from .context import VMContext

logger = logging.getLogger(__name__)

LAST_PARENT = "$LAST_PARENT"


def _resolve_parent_ref(ctx: VMContext, ref: str) -> str:
    """Resolve $LAST_PARENT to the node_id from the most recent BUILD_PARENT."""
    if ref == LAST_PARENT:
        if not ctx.built_parents:
            return "__UNRESOLVED__"
        return ctx.built_parents[-1].get("node_id", "__UNRESOLVED__")
    return ref


# ── Memory Traversal Handlers ────────────────────────────────────


def handle_mem_ptr(ctx: VMContext, inst: Instruction) -> Any:
    """MEM_PTR <query> → closest parent node_id via semantic search."""
    query = " ".join(inst.args) if inst.args else ""
    if not ctx.encoder or not ctx.index:
        return None
    vec = ctx.encoder.encode(query)
    results = ctx.index.search(vec, top_k=1)
    if results:
        node_id = results[0][0]
        ctx.side_effects.append(f"MEM_PTR: found {node_id}")
        return node_id
    return None


def handle_mem_read(ctx: VMContext, inst: Instruction) -> Any:
    """MEM_READ <node_id> → node summary text."""
    if not inst.args or not ctx.store:
        return None
    node_id = inst.args[0]
    node = ctx.store.get_parent_node(node_id)
    if node:
        return node.summary
    child = ctx.store.get_child_node(node_id)
    if child:
        return child.summary
    return None


def handle_mem_next(ctx: VMContext, inst: Instruction) -> Any:
    """MEM_NEXT <node_id> → next sibling child node_id."""
    if not inst.args or not ctx.store:
        return None
    node_id = inst.args[0]
    child = ctx.store.get_child_node(node_id)
    if not child:
        return None
    siblings = ctx.store.get_child_nodes_for_parent(child.parent_node_id)
    for i, sib in enumerate(siblings):
        if sib.node_id == node_id and i + 1 < len(siblings):
            return siblings[i + 1].node_id
    return None


def handle_mem_prev(ctx: VMContext, inst: Instruction) -> Any:
    """MEM_PREV <node_id> → previous sibling child node_id."""
    if not inst.args or not ctx.store:
        return None
    node_id = inst.args[0]
    child = ctx.store.get_child_node(node_id)
    if not child:
        return None
    siblings = ctx.store.get_child_nodes_for_parent(child.parent_node_id)
    for i, sib in enumerate(siblings):
        if sib.node_id == node_id and i > 0:
            return siblings[i - 1].node_id
    return None


def handle_mem_parent(ctx: VMContext, inst: Instruction) -> Any:
    """MEM_PARENT <node_id> → parent node_id."""
    if not inst.args or not ctx.store:
        return None
    node_id = inst.args[0]
    child = ctx.store.get_child_node(node_id)
    if child:
        return child.parent_node_id
    return None


def handle_mem_jmp(ctx: VMContext, inst: Instruction) -> Any:
    """MEM_JMP <node_id> → context jump (load node into accumulator)."""
    if not inst.args or not ctx.store:
        return None
    node_id = inst.args[0]
    node = ctx.store.get_parent_node(node_id)
    if node:
        ctx.side_effects.append(f"MEM_JMP: jumped to {node_id}")
        return node.summary
    child = ctx.store.get_child_node(node_id)
    if child:
        ctx.side_effects.append(f"MEM_JMP: jumped to child {node_id}")
        return child.summary
    return None


# ── Dream / Consolidation Handlers ───────────────────────────────


def handle_extract_constraint(ctx: VMContext, inst: Instruction) -> Any:
    """EXTRACT_CONSTRAINT <trace_id> <description>"""
    if len(inst.args) < 2:
        return None
    trace_id = inst.args[0]
    description = " ".join(inst.args[1:])
    ctx.constraints.append((trace_id, description))
    ctx.side_effects.append(f"EXTRACT_CONSTRAINT: {description[:50]}")
    return description


def handle_mark_critical(ctx: VMContext, inst: Instruction) -> Any:
    """MARK_CRITICAL <trace_id> <action_index>"""
    if len(inst.args) < 2:
        return None
    trace_id = inst.args[0]
    try:
        action_index = int(inst.args[1])
    except ValueError:
        return None
    ctx.critical_indices.append((trace_id, action_index))
    ctx.side_effects.append(f"MARK_CRITICAL: {trace_id}[{action_index}]")
    return action_index


def handle_mark_noise(ctx: VMContext, inst: Instruction) -> Any:
    """MARK_NOISE <trace_id> <action_index>"""
    if len(inst.args) < 2:
        return None
    trace_id = inst.args[0]
    try:
        action_index = int(inst.args[1])
    except ValueError:
        return None
    ctx.noise_indices.append((trace_id, action_index))
    ctx.side_effects.append(f"MARK_NOISE: {trace_id}[{action_index}]")
    return action_index


def handle_build_parent(ctx: VMContext, inst: Instruction) -> Any:
    """BUILD_PARENT <goal> <summary> <confidence>"""
    if len(inst.args) < 3:
        return None
    goal = inst.args[0]
    summary = inst.args[1]
    try:
        confidence = float(inst.args[2])
    except ValueError:
        confidence = 0.5

    import uuid
    node_id = str(uuid.uuid4())
    parent_data = {
        "node_id": node_id,
        "goal": goal,
        "summary": summary,
        "confidence": confidence,
    }
    ctx.built_parents.append(parent_data)
    ctx.side_effects.append(f"BUILD_PARENT: {goal[:40]}")
    return node_id


def handle_build_child(ctx: VMContext, inst: Instruction) -> Any:
    """BUILD_CHILD <parent_id> <step_idx> <reasoning> <action> <result>"""
    if len(inst.args) < 5:
        return None
    parent_ref = _resolve_parent_ref(ctx, inst.args[0])
    try:
        step_idx = int(inst.args[1])
    except ValueError:
        step_idx = 0
    reasoning = inst.args[2]
    action = inst.args[3]
    result = inst.args[4]

    child_data = {
        "parent_id": parent_ref,
        "step_index": step_idx,
        "reasoning": reasoning,
        "action": action,
        "result": result,
    }
    ctx.built_children.append(child_data)
    ctx.side_effects.append(f"BUILD_CHILD: step {step_idx} of {parent_ref[:12]}")
    return child_data


def handle_lnk_node(ctx: VMContext, inst: Instruction) -> Any:
    """LNK_NODE <source_id> <target_id> <edge_type>"""
    if len(inst.args) < 3:
        return None
    source_id = _resolve_parent_ref(ctx, inst.args[0])
    target_id = _resolve_parent_ref(ctx, inst.args[1])
    edge_type = inst.args[2]
    edge_data = {
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": edge_type,
    }
    ctx.built_edges.append(edge_data)
    ctx.side_effects.append(f"LNK_NODE: {source_id[:8]}→{target_id[:8]}")
    return edge_data


def handle_grp_node(ctx: VMContext, inst: Instruction) -> Any:
    """GRP_NODE <node_a> <node_b> — mark two nodes for grouping/merge."""
    if len(inst.args) < 2:
        return None
    ctx.side_effects.append(f"GRP_NODE: {inst.args[0][:8]} + {inst.args[1][:8]}")
    return (inst.args[0], inst.args[1])


def handle_prn_node(ctx: VMContext, inst: Instruction) -> Any:
    """PRN_NODE <node_id> — prune/remove a node."""
    if not inst.args:
        return None
    ctx.side_effects.append(f"PRN_NODE: {inst.args[0]}")
    return inst.args[0]


# ── System Handlers ──────────────────────────────────────────────


def handle_nop(ctx: VMContext, inst: Instruction) -> Any:
    """NOP — no operation."""
    return None


def handle_yield(ctx: VMContext, inst: Instruction) -> Any:
    """YIELD <message> — append message to output buffer."""
    message = " ".join(inst.args) if inst.args else ""
    ctx.output.append(message)
    return message


def handle_halt(ctx: VMContext, inst: Instruction) -> Any:
    """HALT — stop VM execution (handled by dispatch loop, not here)."""
    return None


# ── Handler Registry ─────────────────────────────────────────────

HANDLER_REGISTRY: dict[Opcode, Any] = {
    # Memory traversal
    Opcode.MEM_PTR:    handle_mem_ptr,
    Opcode.MEM_READ:   handle_mem_read,
    Opcode.MEM_NEXT:   handle_mem_next,
    Opcode.MEM_PREV:   handle_mem_prev,
    Opcode.MEM_PARENT: handle_mem_parent,
    Opcode.MEM_JMP:    handle_mem_jmp,
    # Dream / consolidation
    Opcode.EXTRACT_CONSTRAINT: handle_extract_constraint,
    Opcode.MARK_CRITICAL:      handle_mark_critical,
    Opcode.MARK_NOISE:         handle_mark_noise,
    Opcode.BUILD_PARENT:       handle_build_parent,
    Opcode.BUILD_CHILD:        handle_build_child,
    Opcode.LNK_NODE:           handle_lnk_node,
    Opcode.GRP_NODE:           handle_grp_node,
    Opcode.PRN_NODE:           handle_prn_node,
    # System
    Opcode.NOP:   handle_nop,
    Opcode.YIELD: handle_yield,
    Opcode.HALT:  handle_halt,
}
