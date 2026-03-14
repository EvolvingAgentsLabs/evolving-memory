"""Instruction → text string serializer (for testing/debugging/logging)."""

from __future__ import annotations

from .opcodes import Instruction, Program


def serialize_instruction(instruction: Instruction) -> str:
    """Serialize a single Instruction back to text-assembly format."""
    parts = [instruction.opcode.name]
    for arg in instruction.args:
        if " " in arg or not arg:
            parts.append(f'"{arg}"')
        else:
            parts.append(arg)
    return " ".join(parts)


def serialize_program(program: Program) -> str:
    """Serialize a Program back to multi-line text-assembly."""
    return "\n".join(serialize_instruction(i) for i in program.instructions)
