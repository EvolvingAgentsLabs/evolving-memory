"""Text-assembly parser with multi-mode fallback (RoClaw pattern)."""

from __future__ import annotations

import re
import shlex

from .opcodes import Instruction, Opcode, Program, OPCODE_BY_NAME


class InstructionParser:
    """Parses LLM text output into a Program of typed Instructions.

    Fallback strategy (RoClaw pattern):
      1. Primary: shlex.split() — handles quoted strings correctly
      2. Fallback: regex r'("[^"]*"|\\S+)' — more lenient
      3. Final: str.split() — always succeeds, loses multi-word args
    """

    def parse(self, text: str) -> Program:
        """Parse multi-line text into a Program."""
        program = Program(raw_output=text)
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            instruction = self._parse_line(line, line_number, program)
            if instruction is not None:
                program.instructions.append(instruction)
        return program

    def _parse_line(
        self, line: str, line_number: int, program: Program
    ) -> Instruction | None:
        """Parse a single line into an Instruction or record a parse error."""
        tokens = self._tokenize(line)
        if not tokens:
            return None

        opcode_name = tokens[0].upper()
        opcode = OPCODE_BY_NAME.get(opcode_name)
        if opcode is None:
            program.parse_errors.append(
                f"L{line_number}: unknown opcode '{tokens[0]}'"
            )
            return None

        args = tuple(tokens[1:])
        return Instruction(
            opcode=opcode,
            args=args,
            line_number=line_number,
            raw_text=line,
        )

    @staticmethod
    def _tokenize(line: str) -> list[str]:
        """Tokenize a line with multi-mode fallback."""
        # Mode 1: shlex — handles quoted strings
        try:
            return shlex.split(line)
        except ValueError:
            pass

        # Mode 2: regex — more lenient with quotes
        try:
            tokens = re.findall(r'"([^"]*)"|(\S+)', line)
            result = []
            for quoted, unquoted in tokens:
                result.append(quoted if quoted else unquoted)
            if result:
                return result
        except Exception:
            pass

        # Mode 3: str.split — always succeeds
        return line.split()
