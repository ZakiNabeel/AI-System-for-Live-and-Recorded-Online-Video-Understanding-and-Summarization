"""LLM-powered summarization module."""

from .schema import Summary, KeyPoint, DetectedEvent, Chapter, QAPair
from .summarizer import summarize

__all__ = [
    "summarize",
    "Summary",
    "KeyPoint",
    "DetectedEvent",
    "Chapter",
    "QAPair",
]
