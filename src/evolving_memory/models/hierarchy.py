"""Enums defining the CTE type system — hierarchy levels, outcomes, sources, edge types, router paths."""

from enum import IntEnum, StrEnum


class HierarchyLevel(IntEnum):
    """4-level hierarchy matching RoClaw L1–L4."""
    GOAL = 1
    ARCHITECTURE = 2
    TACTICAL = 3
    REACTIVE = 4


class TraceOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ABORTED = "aborted"
    UNKNOWN = "unknown"


class TraceSource(StrEnum):
    REAL_WORLD = "real_world"
    SIM_3D = "sim_3d"
    SIM_2D = "sim_2d"
    DREAM_TEXT = "dream_text"
    AGENT = "agent"
    UNKNOWN_SOURCE = "unknown_source"


class FailureClass(StrEnum):
    """Structured failure categories for dream engine analysis."""
    PHYSICAL_SLIP = "physical_slip"
    MECHANICAL_STALL = "mechanical_stall"
    VLM_HALLUCINATION = "vlm_hallucination"
    LIGHTING_GLARE = "lighting_glare"
    COMMAND_LOST = "command_lost"
    SENSOR_OCCLUSION = "sensor_occlusion"
    TIMEOUT = "timeout"
    LOGIC_ERROR = "logic_error"
    UNKNOWN_FAILURE = "unknown_failure"


class EdgeType(StrEnum):
    NEXT_STEP = "next_step"
    PREVIOUS_STEP = "previous_step"
    IS_CHILD_OF = "is_child_of"
    CAUSAL = "causal"
    CONTEXT_JUMP = "context_jump"


class RouterPath(StrEnum):
    ZERO_SHOT = "zero_shot"
    MEMORY_TRAVERSAL = "memory_traversal"
    CONTEXT_JUMP = "context_jump"
