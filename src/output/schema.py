"""Output formatting data structures."""

from dataclasses import dataclass
from pathlib import Path


from typing import List


@dataclass
class FormatResult:
    """Result of output formatting."""

    markdown: Path
    html: Path
    report_json: Path
    chapters_txt: Path
    report_card: Path
    total_size_bytes: int
    extra_files: List[Path] = None  # domain-specific extra outputs

    def __post_init__(self) -> None:
        if self.extra_files is None:
            self.extra_files = []
