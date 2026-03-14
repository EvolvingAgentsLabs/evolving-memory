"""Tests for the ISA parser, serializer, and opcode definitions."""

import pytest

from evolving_memory.isa.opcodes import Instruction, Opcode, Program, OPCODE_BY_NAME
from evolving_memory.isa.parser import InstructionParser
from evolving_memory.isa.serializer import serialize_instruction, serialize_program


class TestOpcodes:
    def test_all_opcodes_in_name_map(self):
        for op in Opcode:
            assert op.name in OPCODE_BY_NAME
            assert OPCODE_BY_NAME[op.name] is op

    def test_opcode_ranges(self):
        # Memory traversal: 0x10-0x1F
        assert Opcode.MEM_PTR == 0x10
        assert Opcode.MEM_JMP == 0x15
        # Dream: 0x20-0x2F
        assert Opcode.EXTRACT_CONSTRAINT == 0x20
        assert Opcode.PRN_NODE == 0x27
        # System: 0xF0-0xFF
        assert Opcode.NOP == 0xF0
        assert Opcode.HALT == 0xF2

    def test_instruction_is_frozen(self):
        inst = Instruction(opcode=Opcode.NOP, args=())
        with pytest.raises(AttributeError):
            inst.opcode = Opcode.HALT  # type: ignore[misc]


class TestParser:
    def setup_method(self):
        self.parser = InstructionParser()

    def test_parse_simple_opcode(self):
        prog = self.parser.parse("NOP")
        assert len(prog.instructions) == 1
        assert prog.instructions[0].opcode == Opcode.NOP
        assert prog.instructions[0].args == ()

    def test_parse_opcode_with_args(self):
        prog = self.parser.parse("MEM_PTR my_query")
        inst = prog.instructions[0]
        assert inst.opcode == Opcode.MEM_PTR
        assert inst.args == ("my_query",)

    def test_parse_quoted_args(self):
        prog = self.parser.parse('EXTRACT_CONSTRAINT trace123 "Do not retry without backoff"')
        inst = prog.instructions[0]
        assert inst.opcode == Opcode.EXTRACT_CONSTRAINT
        assert inst.args == ("trace123", "Do not retry without backoff")

    def test_parse_build_parent(self):
        prog = self.parser.parse('BUILD_PARENT "implement auth" "Strategy for JWT auth" 0.85')
        inst = prog.instructions[0]
        assert inst.opcode == Opcode.BUILD_PARENT
        assert inst.args == ("implement auth", "Strategy for JWT auth", "0.85")

    def test_parse_build_child_with_last_parent(self):
        prog = self.parser.parse(
            'BUILD_CHILD $LAST_PARENT 0 "Analyze requirements" "read docs" "understood"'
        )
        inst = prog.instructions[0]
        assert inst.opcode == Opcode.BUILD_CHILD
        assert inst.args[0] == "$LAST_PARENT"
        assert inst.args[1] == "0"

    def test_parse_multiline(self):
        text = """
EXTRACT_CONSTRAINT t1 "avoid retries"
MARK_CRITICAL t1 0
MARK_CRITICAL t1 2
HALT
"""
        prog = self.parser.parse(text)
        assert len(prog.instructions) == 4
        assert prog.instructions[0].opcode == Opcode.EXTRACT_CONSTRAINT
        assert prog.instructions[1].opcode == Opcode.MARK_CRITICAL
        assert prog.instructions[2].opcode == Opcode.MARK_CRITICAL
        assert prog.instructions[3].opcode == Opcode.HALT

    def test_skip_comments(self):
        text = """# This is a comment
NOP
// Another comment
HALT
"""
        prog = self.parser.parse(text)
        assert len(prog.instructions) == 2
        assert prog.instructions[0].opcode == Opcode.NOP
        assert prog.instructions[1].opcode == Opcode.HALT

    def test_skip_empty_lines(self):
        text = "\n\nNOP\n\nHALT\n\n"
        prog = self.parser.parse(text)
        assert len(prog.instructions) == 2

    def test_case_insensitive(self):
        prog = self.parser.parse("halt")
        assert len(prog.instructions) == 1
        assert prog.instructions[0].opcode == Opcode.HALT

    def test_unknown_opcode_is_parse_error(self):
        prog = self.parser.parse("FOOBAR arg1 arg2")
        assert len(prog.instructions) == 0
        assert len(prog.parse_errors) == 1
        assert "unknown opcode" in prog.parse_errors[0].lower()

    def test_unknown_opcode_nonfatal(self):
        text = """NOP
FOOBAR arg1
HALT
"""
        prog = self.parser.parse(text)
        assert len(prog.instructions) == 2  # NOP and HALT still parsed
        assert len(prog.parse_errors) == 1

    def test_line_numbers(self):
        text = """NOP
HALT"""
        prog = self.parser.parse(text)
        assert prog.instructions[0].line_number == 1
        assert prog.instructions[1].line_number == 2

    def test_raw_text_preserved(self):
        line = 'EXTRACT_CONSTRAINT t1 "some description"'
        prog = self.parser.parse(line)
        assert prog.instructions[0].raw_text == line

    def test_raw_output_preserved(self):
        text = "NOP\nHALT"
        prog = self.parser.parse(text)
        assert prog.raw_output == text

    def test_all_16_opcodes_roundtrip(self):
        """Every opcode can be parsed from its name."""
        for op in Opcode:
            prog = self.parser.parse(f"{op.name} arg1")
            assert len(prog.instructions) == 1
            assert prog.instructions[0].opcode == op

    def test_yield_with_message(self):
        prog = self.parser.parse('YIELD "Processing complete"')
        inst = prog.instructions[0]
        assert inst.opcode == Opcode.YIELD
        assert inst.args == ("Processing complete",)

    def test_mark_critical_with_index(self):
        prog = self.parser.parse("MARK_CRITICAL trace_abc 3")
        inst = prog.instructions[0]
        assert inst.opcode == Opcode.MARK_CRITICAL
        assert inst.args == ("trace_abc", "3")

    def test_lnk_node(self):
        prog = self.parser.parse("LNK_NODE node_a node_b causal")
        inst = prog.instructions[0]
        assert inst.opcode == Opcode.LNK_NODE
        assert inst.args == ("node_a", "node_b", "causal")

    def test_full_dream_program(self):
        """Parse a realistic dream program."""
        text = """# SWS Phase: failure analysis
EXTRACT_CONSTRAINT trace_001 "Do not retry API calls without exponential backoff"
EXTRACT_CONSTRAINT trace_001 "Do not ignore rate limit headers"
MARK_CRITICAL trace_001 0
MARK_CRITICAL trace_001 2
MARK_CRITICAL trace_001 4
MARK_NOISE trace_001 1
MARK_NOISE trace_001 3
HALT
"""
        prog = self.parser.parse(text)
        assert len(prog.instructions) == 8
        assert len(prog.parse_errors) == 0
        assert prog.instructions[0].opcode == Opcode.EXTRACT_CONSTRAINT
        assert prog.instructions[-1].opcode == Opcode.HALT

    def test_full_rem_program(self):
        """Parse a realistic REM program with BUILD_PARENT and BUILD_CHILD."""
        text = """BUILD_PARENT "implement JWT auth" "Strategy for implementing JWT authentication with HS256" 0.85
BUILD_CHILD $LAST_PARENT 0 "Research JWT spec" "Read RFC 7519" "Understood claims and signing"
BUILD_CHILD $LAST_PARENT 1 "Implement encoder" "Write jwt_utils.py" "200 lines, HS256 + RS256"
BUILD_CHILD $LAST_PARENT 2 "Write tests" "Create test_jwt.py" "8/8 tests passing"
HALT
"""
        prog = self.parser.parse(text)
        assert len(prog.instructions) == 5
        assert prog.instructions[0].opcode == Opcode.BUILD_PARENT
        assert prog.instructions[1].opcode == Opcode.BUILD_CHILD
        assert prog.instructions[1].args[0] == "$LAST_PARENT"


class TestSerializer:
    def test_serialize_simple(self):
        inst = Instruction(opcode=Opcode.NOP, args=())
        assert serialize_instruction(inst) == "NOP"

    def test_serialize_with_args(self):
        inst = Instruction(opcode=Opcode.MEM_PTR, args=("query",))
        assert serialize_instruction(inst) == "MEM_PTR query"

    def test_serialize_quoted_args(self):
        inst = Instruction(
            opcode=Opcode.EXTRACT_CONSTRAINT,
            args=("t1", "multi word description"),
        )
        result = serialize_instruction(inst)
        assert result == 'EXTRACT_CONSTRAINT t1 "multi word description"'

    def test_serialize_empty_arg_gets_quoted(self):
        inst = Instruction(opcode=Opcode.YIELD, args=("",))
        assert serialize_instruction(inst) == 'YIELD ""'

    def test_roundtrip_all_opcodes(self):
        """Serialize → parse round-trip for all opcodes."""
        parser = InstructionParser()
        for op in Opcode:
            inst = Instruction(opcode=op, args=("arg1", "arg two"))
            text = serialize_instruction(inst)
            prog = parser.parse(text)
            assert len(prog.instructions) == 1
            rt = prog.instructions[0]
            assert rt.opcode == op
            assert rt.args == ("arg1", "arg two")

    def test_serialize_program(self):
        prog = Program(instructions=[
            Instruction(opcode=Opcode.NOP, args=()),
            Instruction(opcode=Opcode.HALT, args=()),
        ])
        result = serialize_program(prog)
        assert result == "NOP\nHALT"
