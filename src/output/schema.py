"""Output formatting data structures."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FormatResult:
    """Result of output formatting."""

    markdown: Path
    html: Path
    report_json: Path
    chapters_txt: Path
    report_card: Path
    total_size_bytes: int
