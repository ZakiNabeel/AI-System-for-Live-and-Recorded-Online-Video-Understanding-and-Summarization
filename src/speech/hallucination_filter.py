"""Heuristics for dropping obvious transcript hallucinations."""

from __future__ import annotations

import re

_HALLUCINATION_PATTERNS = (
    r"thanks for watching",
    r"thank you for watching",
    r"please subscribe",
    r"subscribe",
    r"you",
    r"\.",
    r"\.\.\.",
    r"\[music\]",
    r"music",
    r"applause",
)

_PUNCT_RE = re.compile(r"[\s\.\,\!\?\:\;\"'\(\)\[\]\{\}]+")


def normalize_hallucination_text(text: str) -> str:
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def is_hallucination_candidate(text: str) -> bool:
    cleaned = normalize_hallucination_text(text)
    if not cleaned:
        return True
    return any(re.fullmatch(pattern, cleaned) for pattern in _HALLUCINATION_PATTERNS)
