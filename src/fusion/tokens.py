"""Token counting utilities for multiple LLM backends."""

from __future__ import annotations

import logging
from typing import Literal

LOGGER = logging.getLogger(__name__)

# Default fallback token estimates (1 token ≈ 4 chars in English)
DEFAULT_CHARS_PER_TOKEN = 4


def count_tokens(
    text: str,
    backend: Literal["openai", "anthropic"] = "openai",
) -> int:
    """
    Count tokens for a given backend.

    Args:
        text: Input text to count
        backend: LLM backend ("openai" or "anthropic")

    Returns:
        Token count (estimate if backend unavailable)
    """
    if not text:
        return 0

    if backend == "openai":
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model("gpt-4o")
            return len(enc.encode(text))
        except ImportError:
            LOGGER.warning("tiktoken not available, using fallback token estimate")
            return _fallback_token_count(text)
        except Exception as e:
            LOGGER.warning(f"tiktoken error: {e}, using fallback estimate")
            return _fallback_token_count(text)

    elif backend == "anthropic":
        try:
            from anthropic import Anthropic

            client = Anthropic()
            return client.messages.count_tokens(text)
        except Exception as e:
            LOGGER.warning(f"Anthropic token counting failed: {e}, using fallback estimate")
            return _fallback_token_count(text)

    else:
        raise ValueError(f"Unknown backend: {backend}")


def _fallback_token_count(text: str) -> int:
    """Fallback token estimate: 1 token ≈ 4 chars."""
    return max(1, len(text) // DEFAULT_CHARS_PER_TOKEN)
