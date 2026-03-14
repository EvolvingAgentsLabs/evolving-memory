"""CognitiveVM — dispatch loop that executes ISA programs."""

from __future__ import annotations

import logging
from typing import Any

from ..isa.opcodes import Opcode, Program
from .context import VMContext, VMResult
from .handlers import HANDLER_REGISTRY

logger = logging.getLogger(__name__)


class CognitiveVM:
    """Executes a Program against a VMContext.

    Dispatch loop: fetch instruction → dispatch to handler → store result
    in accumulator → log side effect → increment PC.

    Stops on: HALT opcode, max_instructions reached, or end of program.
    """

    def __init__(
        self,
        store: Any = None,
        index: Any = None,
        encoder: Any = None,
        max_instructions: int = 500,
    ) -> None:
        self._store = store
        self._index = index
        self._encoder = encoder
        self._max_instructions = max_instructions

    def execute(self, program: Program) -> VMResult:
        """Execute a parsed Program and return the result."""
        ctx = VMContext(
            store=self._store,
            index=self._index,
            encoder=self._encoder,
            max_instructions=self._max_instructions,
        )
        return self._run(program, ctx)

    def _run(self, program: Program, ctx: VMContext) -> VMResult:
        """Internal dispatch loop."""
        pc = 0
        instructions = program.instructions

        while pc < len(instructions):
            if ctx.instructions_executed >= ctx.max_instructions:
                logger.warning(
                    "VM hit max_instructions limit (%d)", ctx.max_instructions
                )
                return self._build_result(ctx, error="max_instructions limit reached")

            inst = instructions[pc]

            # HALT stops execution
            if inst.opcode == Opcode.HALT:
                ctx.instructions_executed += 1
                break

            handler = HANDLER_REGISTRY.get(inst.opcode)
            if handler is None:
                logger.warning("No handler for opcode %s", inst.opcode.name)
                pc += 1
                ctx.instructions_executed += 1
                continue

            try:
                result = handler(ctx, inst)
                ctx.accumulator = result
            except Exception as exc:
                logger.error(
                    "Handler error at L%d (%s): %s",
                    inst.line_number,
                    inst.opcode.name,
                    exc,
                )
                ctx.side_effects.append(
                    f"ERROR at L{inst.line_number} ({inst.opcode.name}): {exc}"
                )

            ctx.instructions_executed += 1
            pc += 1

        return self._build_result(ctx)

    @staticmethod
    def _build_result(ctx: VMContext, error: str | None = None) -> VMResult:
        """Build a VMResult from the current context state."""
        return VMResult(
            success=error is None,
            instructions_executed=ctx.instructions_executed,
            output=list(ctx.output),
            side_effects=list(ctx.side_effects),
            error=error,
            critical_indices=list(ctx.critical_indices),
            noise_indices=list(ctx.noise_indices),
            constraints=list(ctx.constraints),
            built_parents=list(ctx.built_parents),
            built_children=list(ctx.built_children),
            built_edges=list(ctx.built_edges),
            accumulator=ctx.accumulator,
        )
