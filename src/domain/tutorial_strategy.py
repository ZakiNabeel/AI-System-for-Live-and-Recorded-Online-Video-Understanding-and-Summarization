"""Tutorial-strategy domain profile — workflow extraction from how-to videos."""

from __future__ import annotations

import json
from pathlib import Path

from .base import fmt_time
from .pseudocode import is_programming_tutorial, render_python_skeleton

_PROMPTS = Path(__file__).parent.parent / "llm" / "prompts" / "domains"


class TutorialStrategyProfile:
    name = "tutorial-strategy"
    description = "How-to tutorials: extracts ordered steps and generates workflow files."

    def chunk_prompt_addendum(self) -> str:
        return (_PROMPTS / "tutorial_strategy.txt").read_text(encoding="utf-8")

    def global_prompt_addendum(self) -> str:
        return ""

    def extra_output_schema(self) -> dict | None:
        return {
            "task": {"type": "string"},
            "prerequisites": {"type": "array", "items": {"type": "string"}},
            "steps": {"type": "array"},
            "decision_points": {"type": "array"},
        }

    def post_process(self, summary: dict, fused: dict, output_dir: Path) -> list[Path]:
        """Write strategy.json, strategy.md, and optionally strategy.py."""
        created: list[Path] = []
        extras = {**summary, **summary.get("domain_extras", {})}

        task = extras.get("task", "")
        prerequisites = extras.get("prerequisites", [])
        steps = extras.get("steps", [])
        decision_points = extras.get("decision_points", [])
        video_url = fused.get("source_url", fused.get("url", ""))

        # Normalise steps — guard against empty/malformed
        if not isinstance(steps, list):
            steps = []
        if not isinstance(decision_points, list):
            decision_points = []

        # strategy.json
        strategy_data = {
            "task": task,
            "prerequisites": prerequisites,
            "steps": steps,
            "decision_points": decision_points,
        }
        json_path = output_dir / "strategy.json"
        json_path.write_text(
            json.dumps(strategy_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        created.append(json_path)

        # strategy.md
        md_path = _write_strategy_md(output_dir, task, prerequisites, steps, decision_points)
        created.append(md_path)

        # strategy.py — only for programming tutorials
        if steps and is_programming_tutorial(task, steps):
            code = render_python_skeleton(steps, task, video_url, decision_points)
            py_path = output_dir / "strategy.py"
            py_path.write_text(code, encoding="utf-8")
            created.append(py_path)

        return created


def _write_strategy_md(
    output_dir: Path,
    task: str,
    prerequisites: list,
    steps: list,
    decision_points: list,
) -> Path:
    lines: list[str] = [f"# Tutorial Strategy: {task}\n"]

    if prerequisites:
        lines.append("## Prerequisites\n")
        for p in prerequisites:
            lines.append(f"- {p}")
        lines.append("")

    if steps:
        lines.append("## Steps\n")
        for s in steps:
            if not isinstance(s, dict):
                continue
            ts = fmt_time(float(s.get("timestamp", 0)))
            action = s.get("action", "")
            outcome = s.get("expected_outcome", "")
            params = s.get("parameters", {}) or {}
            param_str = (
                ", ".join(f"`{k}={v}`" for k, v in params.items()) if params else ""
            )
            lines.append(f"### Step {s.get('step_number', '?')}: {action} [{ts}]")
            if param_str:
                lines.append(f"- Parameters: {param_str}")
            if outcome:
                lines.append(f"- Expected: {outcome}")
            lines.append("")

    if decision_points:
        lines.append("## Decision Points\n")
        for dp in decision_points:
            if not isinstance(dp, dict):
                continue
            lines.append(
                f"- **Step {dp.get('step_number', '?')}**: "
                f"If `{dp.get('condition', '')}` "
                f"→ {dp.get('branch_yes', '')} "
                f"| else → {dp.get('branch_no', '')}"
            )
        lines.append("")

    md_path = output_dir / "strategy.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path
