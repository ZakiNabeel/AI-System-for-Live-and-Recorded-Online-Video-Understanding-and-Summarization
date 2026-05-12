"""Education domain profile — lectures, courses, tutorials."""

from __future__ import annotations

from pathlib import Path

from .base import fmt_time

_PROMPTS = Path(__file__).parent.parent / "llm" / "prompts" / "domains"


class EducationProfile:
    name = "education"
    description = "Lecture, course, or educational tutorial content."

    def chunk_prompt_addendum(self) -> str:
        return (_PROMPTS / "education.txt").read_text(encoding="utf-8")

    def global_prompt_addendum(self) -> str:
        return ""

    def extra_output_schema(self) -> dict | None:
        return {
            "learning_objectives": {"type": "array", "items": {"type": "string"}},
            "worked_examples": {"type": "array"},
            "definitions": {"type": "array"},
        }

    def post_process(self, summary: dict, fused: dict, output_dir: Path) -> list[Path]:
        """Write study_notes.md from extracted education fields."""
        # Collect domain extras from wherever the LLM placed them
        extras = {**summary, **summary.get("domain_extras", {})}

        lines: list[str] = ["# Study Notes\n"]

        objectives = extras.get("learning_objectives", [])
        if objectives:
            lines.append("## Learning Objectives\n")
            for obj in objectives:
                lines.append(f"- {obj}")
            lines.append("")

        definitions = extras.get("definitions", [])
        if definitions:
            lines.append("## Key Definitions\n")
            for d in definitions:
                term = d.get("term", "") if isinstance(d, dict) else str(d)
                defn = d.get("definition", "") if isinstance(d, dict) else ""
                ts = fmt_time(float(d.get("timestamp", 0))) if isinstance(d, dict) else ""
                ts_str = f" [{ts}]" if ts else ""
                lines.append(f"**{term}**{ts_str}: {defn}")
            lines.append("")

        examples = extras.get("worked_examples", [])
        if examples:
            lines.append("## Worked Examples\n")
            for ex in examples:
                if not isinstance(ex, dict):
                    continue
                ts = fmt_time(float(ex.get("timestamp", 0)))
                lines.append(f"### [{ts}] {ex.get('problem_statement', '')}")
                for step in ex.get("key_steps", []):
                    lines.append(f"- {step}")
                if ex.get("final_answer"):
                    lines.append(f"\n**Answer:** {ex['final_answer']}")
                lines.append("")

        key_points = summary.get("key_points", [])
        if key_points:
            lines.append("## Key Points\n")
            for kp in key_points:
                if isinstance(kp, dict):
                    ts = fmt_time(float(kp.get("timestamp", 0)))
                    lines.append(f"- [{ts}] {kp.get('text', '')}")
                else:
                    lines.append(f"- {kp}")
            lines.append("")

        out_path = output_dir / "study_notes.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return [out_path]
