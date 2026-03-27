"""Tests for the Cognitive VM — handlers, dispatch loop, and safety limits."""

import pytest

from evolving_memory.isa.opcodes import Instruction, Opcode, Program
from evolving_memory.isa.parser import InstructionParser
from evolving_memory.vm.context import VMContext, VMResult
from evolving_memory.vm.handlers import (
    HANDLER_REGISTRY,
    handle_build_child,
    handle_build_parent,
    handle_extract_constraint,
    handle_halt,
    handle_mark_critical,
    handle_mark_noise,
    handle_nop,
    handle_yield,
)
from evolving_memory.vm.machine import CognitiveVM


class TestHandlerRegistry:
    def test_all_opcodes_have_handlers(self):
        for op in Opcode:
            assert op in HANDLER_REGISTRY, f"Missing handler for {op.name}"


class TestSystemHandlers:
    def test_nop(self):
        ctx = VMContext()
        result = handle_nop(ctx, Instruction(Opcode.NOP, ()))
        assert result is None

    def test_yield(self):
        ctx = VMContext()
        result = handle_yield(
            ctx, Instruction(Opcode.YIELD, ("hello", "world"))
        )
        assert result == "hello world"
        assert ctx.output == ["hello world"]

    def test_yield_empty(self):
        ctx = VMContext()
        handle_yield(ctx, Instruction(Opcode.YIELD, ()))
        assert ctx.output == [""]

    def test_halt(self):
        ctx = VMContext()
        result = handle_halt(ctx, Instruction(Opcode.HALT, ()))
        assert result is None


class TestDreamHandlers:
    def test_extract_constraint(self):
        ctx = VMContext()
        inst = Instruction(
            Opcode.EXTRACT_CONSTRAINT,
            ("trace_001", "Do not retry without backoff"),
        )
        result = handle_extract_constraint(ctx, inst)
        assert result == "Do not retry without backoff"
        assert len(ctx.constraints) == 1
        assert ctx.constraints[0] == ("trace_001", "Do not retry without backoff", "")

    def test_extract_constraint_with_failure_class(self):
        ctx = VMContext()
        inst = Instruction(
            Opcode.EXTRACT_CONSTRAINT,
            ("trace_001", "Do not retry without backoff", "logic_error"),
        )
        result = handle_extract_constraint(ctx, inst)
        assert result == "Do not retry without backoff"
        assert len(ctx.constraints) == 1
        assert ctx.constraints[0] == ("trace_001", "Do not retry without backoff", "logic_error")

    def test_extract_constraint_multi_word_desc_with_failure_class(self):
        ctx = VMContext()
        inst = Instruction(
            Opcode.EXTRACT_CONSTRAINT,
            ("trace_001", "Do", "not", "retry", "without", "backoff", "mechanical_stall"),
        )
        result = handle_extract_constraint(ctx, inst)
        assert result == "Do not retry without backoff"
        assert ctx.constraints[0][2] == "mechanical_stall"

    def test_extract_constraint_too_few_args(self):
        ctx = VMContext()
        inst = Instruction(Opcode.EXTRACT_CONSTRAINT, ("trace_001",))
        result = handle_extract_constraint(ctx, inst)
        assert result is None
        assert len(ctx.constraints) == 0

    def test_mark_critical(self):
        ctx = VMContext()
        inst = Instruction(Opcode.MARK_CRITICAL, ("trace_001", "2"))
        result = handle_mark_critical(ctx, inst)
        assert result == 2
        assert ctx.critical_indices == [("trace_001", 2)]

    def test_mark_critical_invalid_index(self):
        ctx = VMContext()
        inst = Instruction(Opcode.MARK_CRITICAL, ("trace_001", "abc"))
        result = handle_mark_critical(ctx, inst)
        assert result is None

    def test_mark_noise(self):
        ctx = VMContext()
        inst = Instruction(Opcode.MARK_NOISE, ("trace_001", "1"))
        result = handle_mark_noise(ctx, inst)
        assert result == 1
        assert ctx.noise_indices == [("trace_001", 1)]

    def test_build_parent(self):
        ctx = VMContext()
        inst = Instruction(
            Opcode.BUILD_PARENT,
            ("implement auth", "Strategy for JWT auth", "0.85"),
        )
        result = handle_build_parent(ctx, inst)
        assert result is not None  # node_id
        assert len(ctx.built_parents) == 1
        assert ctx.built_parents[0]["goal"] == "implement auth"
        assert ctx.built_parents[0]["summary"] == "Strategy for JWT auth"
        assert ctx.built_parents[0]["confidence"] == 0.85

    def test_build_parent_invalid_confidence(self):
        ctx = VMContext()
        inst = Instruction(
            Opcode.BUILD_PARENT,
            ("goal", "summary", "high"),
        )
        handle_build_parent(ctx, inst)
        assert ctx.built_parents[0]["confidence"] == 0.5  # default

    def test_build_child_with_last_parent(self):
        ctx = VMContext()
        # First build a parent
        handle_build_parent(
            ctx,
            Instruction(Opcode.BUILD_PARENT, ("goal", "summary", "0.9")),
        )
        parent_id = ctx.built_parents[0]["node_id"]

        # Now build a child referencing $LAST_PARENT
        inst = Instruction(
            Opcode.BUILD_CHILD,
            ("$LAST_PARENT", "0", "Analyze reqs", "read docs", "understood"),
        )
        result = handle_build_child(ctx, inst)
        assert result is not None
        assert result["parent_id"] == parent_id
        assert result["step_index"] == 0
        assert result["reasoning"] == "Analyze reqs"
        assert result["action"] == "read docs"
        assert result["result"] == "understood"

    def test_build_child_unresolved_parent(self):
        ctx = VMContext()
        inst = Instruction(
            Opcode.BUILD_CHILD,
            ("$LAST_PARENT", "0", "reason", "action", "result"),
        )
        result = handle_build_child(ctx, inst)
        assert result["parent_id"] == "__UNRESOLVED__"

    def test_build_child_explicit_parent_id(self):
        ctx = VMContext()
        inst = Instruction(
            Opcode.BUILD_CHILD,
            ("node_abc", "1", "reason", "action", "result"),
        )
        result = handle_build_child(ctx, inst)
        assert result["parent_id"] == "node_abc"


class TestCognitiveVM:
    def test_empty_program(self):
        vm = CognitiveVM()
        prog = Program(instructions=[])
        result = vm.execute(prog)
        assert result.success is True
        assert result.instructions_executed == 0

    def test_nop_halt(self):
        vm = CognitiveVM()
        prog = Program(instructions=[
            Instruction(Opcode.NOP, ()),
            Instruction(Opcode.HALT, ()),
        ])
        result = vm.execute(prog)
        assert result.success is True
        assert result.instructions_executed == 2

    def test_yield_collects_output(self):
        vm = CognitiveVM()
        prog = Program(instructions=[
            Instruction(Opcode.YIELD, ("hello",)),
            Instruction(Opcode.YIELD, ("world",)),
            Instruction(Opcode.HALT, ()),
        ])
        result = vm.execute(prog)
        assert result.output == ["hello", "world"]

    def test_max_instructions_limit(self):
        vm = CognitiveVM(max_instructions=3)
        # 5 NOPs, no HALT — should stop at 3
        prog = Program(instructions=[
            Instruction(Opcode.NOP, ()),
            Instruction(Opcode.NOP, ()),
            Instruction(Opcode.NOP, ()),
            Instruction(Opcode.NOP, ()),
            Instruction(Opcode.NOP, ()),
        ])
        result = vm.execute(prog)
        assert result.success is False
        assert result.instructions_executed == 3
        assert "max_instructions" in result.error

    def test_halt_stops_early(self):
        vm = CognitiveVM()
        prog = Program(instructions=[
            Instruction(Opcode.NOP, ()),
            Instruction(Opcode.HALT, ()),
            Instruction(Opcode.NOP, ()),  # should not execute
        ])
        result = vm.execute(prog)
        assert result.instructions_executed == 2  # NOP + HALT

    def test_full_dream_program(self):
        """Execute a realistic SWS program."""
        parser = InstructionParser()
        prog = parser.parse("""
EXTRACT_CONSTRAINT trace_001 "Do not retry without backoff" logic_error
EXTRACT_CONSTRAINT trace_001 "Do not ignore rate limit headers"
MARK_CRITICAL trace_001 0
MARK_CRITICAL trace_001 2
MARK_NOISE trace_001 1
HALT
""")
        vm = CognitiveVM()
        result = vm.execute(prog)
        assert result.success is True
        assert len(result.constraints) == 2
        assert result.constraints[0][2] == "logic_error"
        assert result.constraints[1][2] == ""  # no failure_class
        assert len(result.critical_indices) == 2
        assert len(result.noise_indices) == 1

    def test_full_rem_program(self):
        """Execute a realistic REM program with parent + children."""
        parser = InstructionParser()
        prog = parser.parse("""
BUILD_PARENT "implement JWT auth" "Strategy for JWT authentication" 0.85
BUILD_CHILD $LAST_PARENT 0 "Research spec" "Read RFC 7519" "Understood claims"
BUILD_CHILD $LAST_PARENT 1 "Implement encoder" "Write jwt_utils.py" "200 lines"
BUILD_CHILD $LAST_PARENT 2 "Write tests" "Create test_jwt.py" "8/8 passing"
HALT
""")
        vm = CognitiveVM()
        result = vm.execute(prog)
        assert result.success is True
        assert len(result.built_parents) == 1
        assert len(result.built_children) == 3
        assert result.built_parents[0]["goal"] == "implement JWT auth"
        # Children should all reference the parent
        parent_id = result.built_parents[0]["node_id"]
        for child in result.built_children:
            assert child["parent_id"] == parent_id

    def test_accumulator_tracks_last_result(self):
        vm = CognitiveVM()
        prog = Program(instructions=[
            Instruction(Opcode.YIELD, ("first",)),
            Instruction(Opcode.YIELD, ("second",)),
            Instruction(Opcode.HALT, ()),
        ])
        result = vm.execute(prog)
        assert result.accumulator == "second"

    def test_side_effects_logged(self):
        parser = InstructionParser()
        prog = parser.parse("""
EXTRACT_CONSTRAINT t1 "avoid retries"
MARK_CRITICAL t1 0
HALT
""")
        vm = CognitiveVM()
        result = vm.execute(prog)
        assert len(result.side_effects) >= 2

    def test_parse_errors_dont_prevent_execution(self):
        parser = InstructionParser()
        prog = parser.parse("""
NOP
BOGUS_OPCODE arg1
HALT
""")
        assert len(prog.parse_errors) == 1
        vm = CognitiveVM()
        result = vm.execute(prog)
        assert result.success is True
        assert result.instructions_executed == 2  # NOP + HALT
