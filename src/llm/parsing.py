"""Robust JSON parsing for LLM outputs."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

LOGGER = logging.getLogger(__name__)


class LLMOutputParseError(Exception):
    """Raised when LLM output cannot be parsed as JSON."""

    def __init__(self, text: str, original_error: Exception | None = None):
        self.text = text
        self.original_error = original_error
        super().__init__(f"Failed to parse LLM output: {original_error}")


def strip_and_parse_json(text: str) -> dict[str, Any]:
    """
    Parse JSON from LLM output, recovering from common formatting issues.

    Handles:
    - Direct JSON: {"key": "value"}
    - Markdown fences: ```json\n{...}\n```
    - Extra prose before/after JSON
    - Trailing commas (to some extent)

    Args:
        text: Raw LLM output

    Returns:
        Parsed JSON dict

    Raises:
        LLMOutputParseError: If JSON cannot be extracted
    """
    if not text:
        raise LLMOutputParseError(text, ValueError("Empty text"))

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try removing markdown fences
    text_clean = text.strip()
    if text_clean.startswith("```"):
        # Remove opening fence (with optional language specifier)
        match = re.match(r"```(?:json)?\s*\n", text_clean)
        if match:
            text_clean = text_clean[match.end() :]
        # Remove closing fence
        if text_clean.endswith("```"):
            text_clean = text_clean[:-3].rstrip()

    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        pass

    # Find the largest {...} block
    braces = []
    max_start = 0
    max_depth = 0
    max_len = 0

    for i, char in enumerate(text_clean):
        if char == "{":
            braces.append(i)
        elif char == "}":
            if braces:
                start = braces.pop()
                length = i - start + 1
                if length > max_len:
                    max_len = length
                    max_start = start

    if max_len > 0:
        try:
            json_text = text_clean[max_start : max_start + max_len]
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            LOGGER.debug(f"Failed to parse extracted JSON block: {e}")

    # Last resort: try to fix trailing commas
    try:
        text_fixed = re.sub(r",\s*([}\]])", r"\1", text_clean)
        return json.loads(text_fixed)
    except json.JSONDecodeError as e:
        raise LLMOutputParseError(text, e)


def validate_timestamps(payload: dict[str, Any], duration: float) -> dict[str, Any]:
    """
    Validate and clip timestamps in parsed output.

    Removes events with timestamps outside [0, duration].

    Args:
        payload: Parsed JSON dict
        duration: Valid time range end

    Returns:
        Cleaned payload
    """
    if not isinstance(payload, dict):
        return payload

    # Clean key_points
    if "key_points" in payload and isinstance(payload["key_points"], list):
        payload["key_points"] = [
            kp
            for kp in payload["key_points"]
            if isinstance(kp, dict) and 0 <= float(kp.get("timestamp", 0)) <= duration
        ]

    # Clean events
    if "events" in payload and isinstance(payload["events"], list):
        payload["events"] = [
            evt
            for evt in payload["events"]
            if isinstance(evt, dict) and 0 <= float(evt.get("timestamp", 0)) <= duration
        ]

    # Clean merged_key_points (multi-pass output)
    if "merged_key_points" in payload and isinstance(payload["merged_key_points"], list):
        payload["merged_key_points"] = [
            kp
            for kp in payload["merged_key_points"]
            if isinstance(kp, dict) and 0 <= float(kp.get("timestamp", 0)) <= duration
        ]

    # Clean merged_events (multi-pass output)
    if "merged_events" in payload and isinstance(payload["merged_events"], list):
        payload["merged_events"] = [
            evt
            for evt in payload["merged_events"]
            if isinstance(evt, dict) and 0 <= float(evt.get("timestamp", 0)) <= duration
        ]

    # Clean chapters
    if "chapters" in payload and isinstance(payload["chapters"], list):
        payload["chapters"] = [
            ch
            for ch in payload["chapters"]
            if isinstance(ch, dict)
            and 0 <= float(ch.get("t_start", 0)) <= duration
            and 0 <= float(ch.get("t_end", 0)) <= duration
        ]

    # Sort chapters by t_start
    if "chapters" in payload and isinstance(payload["chapters"], list):
        payload["chapters"] = sorted(payload["chapters"], key=lambda ch: float(ch.get("t_start", 0)))

    # Clean Q&A pairs
    if "qa_pairs" in payload and isinstance(payload["qa_pairs"], list):
        payload["qa_pairs"] = [
            qa
            for qa in payload["qa_pairs"]
            if isinstance(qa, dict) and 0 <= float(qa.get("timestamp", 0)) <= duration
        ]

    return payload
