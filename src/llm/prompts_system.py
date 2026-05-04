"""Prompt loading and template system."""

from __future__ import annotations

import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str, **variables) -> str:
    """
    Load a prompt template and substitute variables.

    Args:
        name: Prompt name (e.g., "chunk", "global", "domain_education")
        **variables: Variables to substitute in template

    Returns:
        Rendered prompt

    Raises:
        FileNotFoundError: If prompt not found
    """
    prompt_path = PROMPTS_DIR / f"{name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    template = prompt_path.read_text(encoding="utf-8")

    for key, value in variables.items():
        text = value if isinstance(value, str) else json.dumps(value, default=str, indent=2)
        template = template.replace("{" + key + "}", text)
    return template


SYSTEM_BASE = """You are an expert video analyst and content summarizer. Your task is to analyze video transcripts, 
OCR text, and visual descriptions to produce accurate, well-structured summaries.

Always follow the JSON output format exactly. Output ONLY the JSON object with no additional text, prose, or markdown."""

SYSTEM_CHUNK = SYSTEM_BASE + """

For this task, analyze a chronological segment of a video and extract local insights."""

SYSTEM_GLOBAL = SYSTEM_BASE + """

For this task, synthesize multiple segment summaries into a coherent global summary."""


def get_chunk_prompt(events_json: str) -> tuple[str, str]:
    """
    Get system and user messages for per-chunk summarization.

    Args:
        events_json: JSON string of fused events

    Returns:
        (system_message, user_message)
    """
    try:
        prompt_text = load_prompt("chunk", events_json=events_json)
    except FileNotFoundError:
        LOGGER.warning("Using default chunk prompt")
        prompt_text = f"""Analyze the following video segment and extract key insights in JSON format:

{{
  "local_summary": "<2-3 sentences>",
  "key_points": [
    {{"timestamp": <float>, "text": "<text>", "confidence": "low|medium|high"}},
    ...
  ],
  "events": [
    {{"timestamp": <float>, "event_type": "<type>", "description": "<desc>"}},
    ...
  ]
}}

Events JSON:
{events_json}"""

    return SYSTEM_CHUNK, prompt_text


def get_global_prompt(local_summaries_json: str) -> tuple[str, str]:
    """
    Get system and user messages for global synthesis.

    Args:
        local_summaries_json: JSON string of local summaries

    Returns:
        (system_message, user_message)
    """
    try:
        prompt_text = load_prompt("global", local_summaries_json=local_summaries_json)
    except FileNotFoundError:
        LOGGER.warning("Using default global prompt")
        prompt_text = f"""Synthesize the following local segment summaries into a single coherent global summary:

{{
  "full_summary": "<3-6 paragraphs>",
  "short_summary": "<1-2 sentence pitch>",
  "chapters": [
    {{"t_start": <float>, "t_end": <float>, "title": "<title>"}},
    ...
  ],
  "merged_key_points": [...],
  "merged_events": [...]
}}

Local summaries:
{local_summaries_json}"""

    return SYSTEM_GLOBAL, prompt_text
