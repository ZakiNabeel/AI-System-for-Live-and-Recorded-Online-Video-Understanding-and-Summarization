"""DomainProfile Protocol and shared helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class DomainProfile(Protocol):
    """Contract every domain profile must satisfy."""

    name: str
    description: str

    def chunk_prompt_addendum(self) -> str:
        """Extra text appended to the chunk prompt before the INPUT section."""
        ...

    def global_prompt_addendum(self) -> str:
        """Extra text appended to the global prompt before the INPUT section."""
        ...

    def extra_output_schema(self) -> dict | None:
        """JSON schema fragment for extra LLM fields, or None."""
        ...

    def post_process(self, summary: dict, fused: dict, output_dir: Path) -> list[Path]:
        """Generate domain-specific extra files. Return list of created paths."""
        ...


def fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
