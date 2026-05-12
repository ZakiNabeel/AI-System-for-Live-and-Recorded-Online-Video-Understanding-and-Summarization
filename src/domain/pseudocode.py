"""Generate a Python skeleton from extracted tutorial steps."""

from __future__ import annotations

import ast
import re

from .base import fmt_time

PROGRAMMING_KEYWORDS = {
    "python", "function", "code", "script", "command", "git", "docker",
    "terminal", "bash", "javascript", "typescript", "react", "api", "sql",
    "install", "pip", "npm", "yarn", "class", "import", "module", "debug",
    "programming", "coding", "program", "software", "library", "framework",
    "compile", "run", "execute", "deploy", "server", "database", "query",
}


def is_programming_tutorial(task: str, steps: list[dict]) -> bool:
    """Return True if the tutorial appears to be about programming/coding."""
    text = (task + " " + " ".join(s.get("action", "") for s in steps)).lower()
    return any(kw in text for kw in PROGRAMMING_KEYWORDS)


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:40]


def render_python_skeleton(
    steps: list[dict],
    task: str,
    source_url: str = "",
    decision_points: list[dict] | None = None,
) -> str:
    """
    Render a Python skeleton file with one function per tutorial step.

    Validates output with ast.parse; falls back to a comment-only block on failure.
    """
    decision_points = decision_points or []
    dp_map: dict[int, dict] = {
        dp["step_number"]: dp for dp in decision_points if isinstance(dp, dict)
    }

    header_lines = ['"""']
    header_lines.append(f"Auto-generated workflow skeleton.")
    header_lines.append(f"Task: {task}")
    if source_url:
        header_lines.append(f"Source: {source_url}")
    header_lines += ['"""', "", ""]

    func_lines: list[str] = []
    call_names: list[str] = []

    for s in steps:
        if not isinstance(s, dict):
            continue
        num = s.get("step_number", 0)
        action = s.get("action", "step")
        ts = fmt_time(float(s.get("timestamp", 0)))
        outcome = s.get("expected_outcome", "")
        params = s.get("parameters", {}) or {}

        func_name = f"step_{num:02d}_{_slugify(action)}"
        call_names.append(func_name)

        doc_lines = [f'    """[{ts}] {action}.']
        if params:
            doc_lines.append("")
            doc_lines.append("    Parameters:")
            for k, v in params.items():
                doc_lines.append(f"        {k}: {v}")
        if outcome:
            doc_lines.append("")
            doc_lines.append(f"    Expected: {outcome}")
        doc_lines.append('    """')

        body_lines = ["    # TODO: implement", "    ..."]

        # Add decision branch comment if present
        if num in dp_map:
            dp = dp_map[num]
            body_lines += [
                f"    # Decision: if {dp.get('condition', '')}:",
                f"    #     -> {dp.get('branch_yes', '')}",
                f"    # else:",
                f"    #     -> {dp.get('branch_no', '')}",
            ]

        func_lines.append(f"def {func_name}():")
        func_lines.extend(doc_lines)
        func_lines.extend(body_lines)
        func_lines.append("")
        func_lines.append("")

    main_lines = ["if __name__ == '__main__':"]
    if call_names:
        for name in call_names:
            main_lines.append(f"    {name}()")
    else:
        main_lines.append("    pass")

    code = "\n".join(header_lines + func_lines + main_lines + [""])

    # Validate — fall back to comment block on parse error
    try:
        ast.parse(code)
    except SyntaxError:
        fallback = ["# Auto-generated workflow (syntax validation failed)\n"]
        for s in steps:
            if isinstance(s, dict):
                ts = fmt_time(float(s.get("timestamp", 0)))
                fallback.append(
                    f"# Step {s.get('step_number', '?')} [{ts}]: {s.get('action', '')}"
                )
        code = "\n".join(fallback)

    return code
