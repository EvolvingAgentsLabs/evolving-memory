"""Core ISA definitions — opcodes, instructions, and programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Opcode(IntEnum):
    """Cognitive ISA opcodes for the Evolving Memory VM."""

    # Memory Traversal (0x10-0x1F)
    MEM_PTR    = 0x10  # MEM_PTR <query>              → closest parent node_id
    MEM_READ   = 0x11  # MEM_READ <node_id>            → node summary
    MEM_NEXT   = 0x12  # MEM_NEXT <node_id>            → next child node_id
    MEM_PREV   = 0x13  # MEM_PREV <node_id>            → previous child node_id
    MEM_PARENT = 0x14  # MEM_PARENT <node_id>          → parent node_id
    MEM_JMP    = 0x15  # MEM_JMP <node_id>             → context jump

    # Dream / Consolidation (0x20-0x2F)
    EXTRACT_CONSTRAINT = 0x20  # EXTRACT_CONSTRAINT <trace_id> <description>
    MARK_CRITICAL      = 0x21  # MARK_CRITICAL <trace_id> <action_index>
    MARK_NOISE         = 0x22  # MARK_NOISE <trace_id> <action_index>
    BUILD_PARENT       = 0x23  # BUILD_PARENT <goal> <summary> <confidence>
    BUILD_CHILD        = 0x24  # BUILD_CHILD <parent_id> <step_idx> <reasoning> <action> <result>
    LNK_NODE           = 0x25  # LNK_NODE <source_id> <target_id> <edge_type>
    GRP_NODE           = 0x26  # GRP_NODE <node_a> <node_b>
    PRN_NODE           = 0x27  # PRN_NODE <node_id>

    # System (0xF0-0xFF)
    NOP   = 0xF0
    YIELD = 0xF1  # YIELD <message>
    HALT  = 0xF2


# Reverse lookup: opcode name (uppercase) → Opcode enum member
OPCODE_BY_NAME: dict[str, Opcode] = {op.name: op for op in Opcode}


@dataclass(frozen=True)
class Instruction:
    """A single parsed instruction."""
    opcode: Opcode
    args: tuple[str, ...]
    line_number: int = 0
    raw_text: str = ""


@dataclass
class Program:
    """A parsed program — a sequence of instructions with optional parse errors."""
    instructions: list[Instruction] = field(default_factory=list)
    raw_output: str = ""
    parse_errors: list[str] = field(default_factory=list)
