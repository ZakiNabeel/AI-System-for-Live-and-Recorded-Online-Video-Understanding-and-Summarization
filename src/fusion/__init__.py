"""Multimodal event fusion module.

Merges aligned transcript and visual extraction into time-ordered multimodal events
ready for LLM summarization.
"""

from .fuser import fuse
from .schema import FusedDocument, FusedEvent

__all__ = [
    "fuse",
    "FusedDocument",
    "FusedEvent",
]
