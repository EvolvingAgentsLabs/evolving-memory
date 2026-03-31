"""Typed LLM IO — ported from claw-code's strict typed content blocks.

Provides structured response wrappers and robust JSON extraction
with multi-stage fallback parsing.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """Base response wrapper for all LLM calls."""
    raw_text: str


class LLMJsonResponse(LLMResponse):
    """Response containing parsed JSON data."""
    data: dict


class LLMProgramResponse(LLMResponse):
    """Response containing ISA program output from dream phases."""
    pass


class LLMParseError(Exception):
    """Raised when LLM output cannot be parsed into the expected format."""

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


def extract_json_robust(text: str) -> dict:
    """Extract JSON from LLM output with multi-stage fallback.

    Stages:
      1. Direct json.loads(text)
      2. Extract from markdown code blocks (```json ... ```)
      3. Extract first {...} or [...] block via brace matching
      4. Raise LLMParseError with raw text attached

    Logs a warning when a fallback stage is used.
    """
    # Stage 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"items": result}
    except (json.JSONDecodeError, ValueError):
        pass

    # Stage 2: markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            logger.warning("JSON extracted from markdown code block (fallback stage 2)")
            if isinstance(result, dict):
                return result
            if isinstance(result, list):
                return {"items": result}
        except (json.JSONDecodeError, ValueError):
            pass

    # Stage 3: find first balanced {...} block
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(brace_start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start : i + 1]
                    try:
                        result = json.loads(candidate)
                        logger.warning("JSON extracted via brace matching (fallback stage 3)")
                        if isinstance(result, dict):
                            return result
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

    raise LLMParseError(
        f"Failed to extract JSON from LLM output ({len(text)} chars)",
        raw_text=text,
    )
