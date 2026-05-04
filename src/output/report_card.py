"""Report card generator for submission grading."""

from pathlib import Path


def render_report_card(output_dir: Path) -> str:
    """
    Generate submission report card showing which deliverables exist.

    Args:
        output_dir: Output directory containing the deliverables

    Returns:
        Markdown report card string
    """
    deliverables = {
        "summary.md": "Markdown report",
        "summary.html": "HTML report",
        "report.json": "Machine-readable JSON",
        "chapters.txt": "YouTube chapter markers",
    }

    lines = [
        "# Submission Report Card",
        "",
        "| Deliverable | Path | Present |",
        "|---|---|---|",
    ]

    for filename, description in deliverables.items():
        path = output_dir / filename
        present = "✅" if filename == "report_card.md" or path.exists() else "❌"
        lines.append(f"| {description} | `{filename}` | {present} |")

    lines.append("")
    lines.append("## Summary")
    lines.append("")

    all_present = all(
        filename == "report_card.md" or (output_dir / filename).exists()
        for filename in deliverables.keys()
    )
    if all_present:
        lines.append("✅ All deliverables present and ready for submission.")
    else:
        missing = [
            filename
            for filename in deliverables.keys()
            if filename != "report_card.md" and not (output_dir / filename).exists()
        ]
        lines.append(f"⚠️  Missing deliverables: {', '.join(missing)}")

    return "\n".join(lines)
