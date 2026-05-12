"""Medical domain profile — clinical education, conference talks, lectures."""

from __future__ import annotations

from pathlib import Path

from .base import fmt_time

_PROMPTS = Path(__file__).parent.parent / "llm" / "prompts" / "domains"

_DISCLAIMER = (
    "> **DISCLAIMER:** This is an AI-generated educational summary. "
    "It is **not** for diagnostic or treatment decisions. "
    "Always verify with primary sources and qualified clinicians.\n\n"
)


class MedicalProfile:
    name = "medical"
    description = "Medical education, clinical lectures, and conference talks."

    def chunk_prompt_addendum(self) -> str:
        return (_PROMPTS / "medical.txt").read_text(encoding="utf-8")

    def global_prompt_addendum(self) -> str:
        return ""

    def extra_output_schema(self) -> dict | None:
        return None

    def post_process(self, summary: dict, fused: dict, output_dir: Path) -> list[Path]:
        """Prepend disclaimer to summary.md and write clinical_notes.md."""
        created: list[Path] = []
        extras = {**summary, **summary.get("domain_extras", {})}

        # Prepend disclaimer to summary.md
        md_path = output_dir / "summary.md"
        if md_path.exists():
            original = md_path.read_text(encoding="utf-8")
            md_path.write_text(_DISCLAIMER + original, encoding="utf-8")

        # Write clinical_notes.md
        lines: list[str] = [_DISCLAIMER, "# Clinical Notes\n"]

        conditions = extras.get("conditions_discussed", [])
        if conditions:
            lines.append("## Conditions Discussed\n")
            for c in conditions:
                if not isinstance(c, dict):
                    lines.append(f"- {c}")
                    continue
                ts = fmt_time(float(c.get("timestamp", 0)))
                lines.append(f"- [{ts}] {c.get('name', '')}")
            lines.append("")

        treatments = extras.get("treatments_mentioned", [])
        if treatments:
            lines.append("## Treatments Mentioned\n")
            for t in treatments:
                if not isinstance(t, dict):
                    lines.append(f"- {t}")
                    continue
                ts = fmt_time(float(t.get("timestamp", 0)))
                lines.append(f"- [{ts}] **{t.get('name', '')}** ({t.get('type', '')})")
            lines.append("")

        evidence = extras.get("evidence_levels", [])
        if evidence:
            lines.append("## Evidence Levels\n")
            for e in evidence:
                if not isinstance(e, dict):
                    continue
                ts = fmt_time(float(e.get("timestamp", 0)))
                lines.append(f"- [{ts}] [{e.get('level', '')}] {e.get('claim', '')}")
            lines.append("")

        clinical_points = extras.get("key_clinical_points", [])
        if clinical_points:
            lines.append("## Key Clinical Points\n")
            for pt in clinical_points:
                lines.append(f"- {pt}")
            lines.append("")

        notes_path = output_dir / "clinical_notes.md"
        notes_path.write_text("\n".join(lines), encoding="utf-8")
        created.append(notes_path)
        return created
