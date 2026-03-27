"""Robotics domain adapter — ported from RoClaw's roclaw_dream_adapter.ts.

Provides robot-specific LLM prompts for bytecode motor traces,
spatial navigation rules, and hardware-aware failure analysis.
"""

from __future__ import annotations


class RoboticsAdapter:
    """Robotics adapter for the dream engine.

    Ported from RoClaw's ``roclaw_dream_adapter.ts`` — provides domain-specific
    system prompts that understand bytecode motor actions, spatial coordinate
    hints, and physical sensor constraints.
    """

    @property
    def domain_name(self) -> str:
        return "robotics"

    def sws_system_prompt(self) -> str:
        return (
            "You are analyzing robot execution traces for a memory consolidation system.\n"
            "The robot uses hex-bytecode motor commands and VLM-powered vision.\n"
            "Output ONLY instructions, one per line. No prose, no explanation.\n"
            "Lines starting with # are comments (optional).\n\n"
            "When extracting constraints from failures, classify each with a failure_class:\n"
            "- physical_slip: wheels slipped, overshooting turns or targets\n"
            "- mechanical_stall: motors commanded but robot didn't move (stuck on obstacle)\n"
            "- vlm_hallucination: VLM described objects/paths that don't exist in the scene\n"
            "- lighting_glare: bright light or shadows caused misperception\n"
            "- command_lost: bytecode frame was dropped or unacknowledged\n"
            "- sensor_occlusion: camera blocked or field of view obstructed\n"
            "- timeout: operation exceeded time limit\n"
            "- logic_error: incorrect reasoning or spatial navigation errors\n"
            "- unknown_failure: cause cannot be determined"
        )

    def rem_system_prompt(self) -> str:
        return (
            "You are abstracting successful robot traces into reusable strategies.\n"
            "The robot uses hex-bytecode motor commands and VLM-powered vision.\n"
            "Output ONLY instructions, one per line. No prose, no explanation.\n"
            "Lines starting with # are comments (optional).\n\n"
            "When building strategy nodes:\n"
            "- Extract spatial navigation rules from bounding box coordinates\n"
            "  (e.g., 'when target bbox center x > 600, TURN_RIGHT proportionally')\n"
            "- Identify motor command patterns (approach, turn, follow-wall)\n"
            "- Note preconditions (camera active, near wall, obstacle detected)\n"
            "- Preserve the mapping from VLM reasoning to bytecode actions"
        )

    def consolidation_context(self) -> str:
        return (
            "This is a robotics domain. When merging strategies, preserve:\n"
            "- Spatial rules mapping bounding box positions to motor actions\n"
            "- Hardware-specific constraints (motor stall thresholds, camera FOV)\n"
            "- Physical navigation patterns (wall-following, doorway approach)"
        )
