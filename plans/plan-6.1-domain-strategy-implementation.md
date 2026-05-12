# Plan 6.1 — Domain-Specific Analysis & Strategy Extraction (Implementation)

> **Self-contained scope.** Build `src/domain/` — a pluggable domain-mode package that re-targets LLM prompts and post-processes summaries into domain-specific deliverables. Covers: education, trading, medical, law, and tutorial-strategy. The tutorial-strategy profile also generates pseudocode (`strategy.py`) and a structured workflow (`strategy.json`/`strategy.md`). This plan converts the design in Plan 5.2 into a concrete, step-by-step implementation guide.

---

## 1. Objective

Implement everything described in Plan 5.2 that has not yet been built:

1. `src/domain/` package with `DomainProfile` Protocol and registry.
2. Five profiles: `education`, `trading`, `medical`, `law`, `tutorial-strategy`.
3. Domain prompt text files under `src/llm/prompts/domains/`.
4. Wire domain parameter into the existing `summarizer.py` so addenda are appended to prompts.
5. Wire domain `post_process()` into `src/output/formatter.py` so extra files are generated.
6. Wire `--domain` flag through `src/pipeline.py` → `src/stages/summarize.py` → `src/stages/format.py`.
7. Tests for each profile and the registry.

---

## 2. Current State

- `src/domain/` directory does **not exist**.
- `src/llm/summarizer.py` already accepts `domain: str | None` parameter (lines 29 and 108/111 in summarizer.py) but silently ignores it — no addendum is injected into prompts.
- `src/llm/prompts_system.py` `get_chunk_prompt` / `get_global_prompt` do **not** prepend domain text.
- `src/llm/prompts/domains/` directory does **not exist**.
- `src/output/formatter.py` does not call any domain post-processor.
- `src/pipeline.py` passes `domain` through to summarize stage but that stage does nothing with it.

---

## 3. Contracts

### 3.1 `DomainProfile` Protocol

```python
# src/domain/base.py
from __future__ import annotations
from pathlib import Path
from typing import Protocol, runtime_checkable

@runtime_checkable
class DomainProfile(Protocol):
    name: str
    description: str

    def chunk_prompt_addendum(self) -> str:
        """Extra instruction block appended to the chunk prompt before the INPUT section."""
        ...

    def global_prompt_addendum(self) -> str:
        """Extra instruction block appended to the global prompt before the INPUT section."""
        ...

    def extra_output_schema(self) -> dict | None:
        """JSON schema fragment describing extra fields the LLM should return. None = no extras."""
        ...

    def post_process(self, summary: dict, fused: dict, output_dir: Path) -> list[Path]:
        """Generate domain-specific extra output files. Return list of created paths."""
        ...
```

### 3.2 Registry

```python
# src/domain/registry.py
from src.domain.education import EducationProfile
from src.domain.trading import TradingProfile
from src.domain.medical import MedicalProfile
from src.domain.law import LawProfile
from src.domain.tutorial_strategy import TutorialStrategyProfile

DOMAINS: dict[str, DomainProfile] = {
    "education":         EducationProfile(),
    "trading":           TradingProfile(),
    "medical":           MedicalProfile(),
    "law":               LawProfile(),
    "tutorial-strategy": TutorialStrategyProfile(),
}


class UnknownDomainError(ValueError):
    def __init__(self, name: str):
        super().__init__(f"Unknown domain '{name}'. Valid: {list(DOMAINS)}")


def get_domain(name: str) -> DomainProfile:
    if name not in DOMAINS:
        raise UnknownDomainError(name)
    return DOMAINS[name]
```

### 3.3 Summarizer Integration

Modify `src/llm/prompts_system.py` — `get_chunk_prompt` and `get_global_prompt` gain an optional `domain_addendum: str` parameter:

```python
def get_chunk_prompt(events_json: str, domain_addendum: str = "") -> tuple[str, str]:
    ...
    if domain_addendum:
        # Insert addendum right before the INPUT section
        prompt_text = prompt_text.replace("INPUT:", f"{domain_addendum.strip()}\n\nINPUT:")
    return SYSTEM_CHUNK, prompt_text
```

Modify `src/llm/summarizer.py` — resolve domain profile and pass addenda:

```python
def _get_addenda(domain: str | None) -> tuple[str, str]:
    """Return (chunk_addendum, global_addendum) for the given domain."""
    if not domain:
        return "", ""
    from src.domain.registry import get_domain
    profile = get_domain(domain)
    return profile.chunk_prompt_addendum(), profile.global_prompt_addendum()
```

Then in `_summarize_singlepass` and `_summarize_multipass`, replace bare calls:
```python
system_chunk, user_chunk = get_chunk_prompt(events_json)
```
with:
```python
chunk_add, global_add = _get_addenda(domain)
system_chunk, user_chunk = get_chunk_prompt(events_json, domain_addendum=chunk_add)
```
and:
```python
system_global, user_global = get_global_prompt(local_summaries_json)
```
with:
```python
system_global, user_global = get_global_prompt(local_summaries_json, domain_addendum=global_add)
```

### 3.4 Formatter Integration

Modify `src/output/formatter.py` — after writing core outputs, invoke domain post-processor:

```python
def format_outputs(..., domain: str | None = None, ...) -> FormatResult:
    ...
    # existing core formatting
    ...
    # Domain extras
    extra_paths: list[Path] = []
    if domain:
        from src.domain.registry import get_domain
        profile = get_domain(domain)
        summary_dict = json.loads(output_dir / "report.json").read_text() ...
        # load summary_dict and fused_dict from their JSON files
        fused_dict = json.loads((fused_path).read_text()) if fused_path else {}
        extra_paths = profile.post_process(summary_dict, fused_dict, output_dir)
        for p in extra_paths:
            LOGGER.info(f"Domain extra: {p}")
    result.extra_files = extra_paths
    return result
```

---

## 4. File Layout to Create

```
src/domain/
  __init__.py              # exports: get_domain, DOMAINS, UnknownDomainError
  base.py                  # DomainProfile Protocol
  registry.py              # DOMAINS dict + get_domain()
  education.py             # EducationProfile
  trading.py               # TradingProfile
  medical.py               # MedicalProfile
  law.py                   # LawProfile
  tutorial_strategy.py     # TutorialStrategyProfile (most complex)
  pseudocode.py            # Python skeleton generator (used by tutorial_strategy)

src/llm/prompts/domains/   # Create this directory
  education.txt
  trading.txt
  medical.txt
  law.txt
  tutorial_strategy.txt

tests/domain/
  __init__.py
  conftest.py              # Shared fixtures (canned summary + fused dicts)
  test_registry.py
  test_education.py
  test_trading.py
  test_medical.py
  test_law.py
  test_tutorial_strategy.py
```

---

## 5. Prompt Files (Content)

### `src/llm/prompts/domains/education.txt`
```
DOMAIN MODE: EDUCATION
Additionally extract from the content:
- "learning_objectives": 3-5 bullet points (strings) of what a student will learn.
- "worked_examples": list of {"timestamp": <float>, "problem_statement": "<str>", "key_steps": ["<str>", ...], "final_answer": "<str>"}.
- "definitions": list of {"term": "<str>", "definition": "<str>", "timestamp": <float>} for new vocabulary introduced.
Include these fields in your JSON output at the top level alongside the required fields.
```

### `src/llm/prompts/domains/trading.txt`
```
DOMAIN MODE: TRADING
Additionally extract:
- "tickers_mentioned": list of ticker symbols (e.g. "AAPL", "BTC"). Only uppercase 1-5 letter symbols.
- "signals": list of {"timestamp": <float>, "ticker": "<str>", "action": "buy|sell|hold|watch", "rationale": "<str>", "timeframe": "<str>"}.
- "indicators": list of {"name": "<str>", "timestamp": <float>, "context": "<str>"} for technical indicators.
- "risk_warnings": list of strings for any caveats or risk disclosures mentioned.
Include these fields in your JSON output at the top level.
```

### `src/llm/prompts/domains/medical.txt`
```
DOMAIN MODE: MEDICAL EDUCATION
Additionally extract:
- "conditions_discussed": list of {"name": "<str>", "timestamp": <float>}.
- "treatments_mentioned": list of {"name": "<str>", "type": "drug|procedure|lifestyle", "timestamp": <float>}.
- "evidence_levels": list of {"claim": "<str>", "level": "RCT|meta-analysis|guideline|expert-opinion", "timestamp": <float>}.
- "key_clinical_points": list of actionable takeaway strings for clinicians.
Include these fields in your JSON output at the top level.
```

### `src/llm/prompts/domains/law.txt`
```
DOMAIN MODE: LEGAL EDUCATION
Additionally extract:
- "cases_cited": list of {"case_name": "<str>", "citation": "<str>", "jurisdiction": "<str>", "timestamp": <float>, "relevance": "<str>"}.
- "statutes": list of {"name": "<str>", "section": "<str>", "timestamp": <float>}.
- "legal_principles": list of {"doctrine": "<str>", "explanation": "<str>", "timestamp": <float>}.
- "arguments": list of {"position": "pro|con|neutral", "statement": "<str>", "timestamp": <float>}.
Include these fields in your JSON output at the top level.
```

### `src/llm/prompts/domains/tutorial_strategy.txt`
```
DOMAIN MODE: TUTORIAL STRATEGY EXTRACTION
Additionally extract a structured workflow from this tutorial video:
- "task": one-sentence description of what the tutorial teaches.
- "prerequisites": list of prerequisite strings (tools, knowledge, software needed).
- "steps": ordered list of {"step_number": <int>, "timestamp": <float>, "action": "<imperative verb phrase>", "parameters": {"<key>": "<value>"}, "expected_outcome": "<str>", "screenshot_frame_index": <int or null>}.
- "decision_points": list of {"step_number": <int>, "condition": "<str>", "branch_yes": "<str>", "branch_no": "<str>"}. Only include if genuine branching exists.
Steps must be atomic: one action per step. Aim for 5-20 steps.
Include these fields in your JSON output at the top level.
```

---

## 6. Profile Implementations

### 6.1 `education.py`

```python
from __future__ import annotations
from pathlib import Path

_PROMPTS = Path(__file__).parent.parent / "llm" / "prompts" / "domains"


class EducationProfile:
    name = "education"
    description = "Lecture, course, or tutorial educational content."

    def chunk_prompt_addendum(self) -> str:
        return (_PROMPTS / "education.txt").read_text(encoding="utf-8")

    def global_prompt_addendum(self) -> str:
        return ""  # chunk addendum is sufficient; global pass just synthesizes

    def extra_output_schema(self) -> dict | None:
        return {
            "learning_objectives": {"type": "array", "items": {"type": "string"}},
            "worked_examples": {"type": "array"},
            "definitions": {"type": "array"},
        }

    def post_process(self, summary: dict, fused: dict, output_dir: Path) -> list[Path]:
        """Write study_notes.md."""
        lines = ["# Study Notes\n"]

        # Learning objectives
        objectives = summary.get("domain_extras", {}).get("learning_objectives",
                     summary.get("learning_objectives", []))
        if objectives:
            lines.append("## Learning Objectives\n")
            for obj in objectives:
                lines.append(f"- {obj}")
            lines.append("")

        # Definitions
        definitions = summary.get("domain_extras", {}).get("definitions",
                      summary.get("definitions", []))
        if definitions:
            lines.append("## Key Definitions\n")
            for d in definitions:
                term = d.get("term", "")
                defn = d.get("definition", "")
                lines.append(f"**{term}**: {defn}")
            lines.append("")

        # Worked examples
        examples = summary.get("domain_extras", {}).get("worked_examples",
                   summary.get("worked_examples", []))
        if examples:
            lines.append("## Worked Examples\n")
            for ex in examples:
                ts = _fmt_time(ex.get("timestamp", 0))
                lines.append(f"### [{ts}] {ex.get('problem_statement', '')}")
                for step in ex.get("key_steps", []):
                    lines.append(f"- {step}")
                if ex.get("final_answer"):
                    lines.append(f"\n**Answer:** {ex['final_answer']}")
                lines.append("")

        # Key points from base summary
        key_points = summary.get("key_points", [])
        if key_points:
            lines.append("## Key Points\n")
            for kp in key_points:
                ts = _fmt_time(kp.get("timestamp", 0))
                lines.append(f"- [{ts}] {kp.get('text', '')}")
            lines.append("")

        out_path = output_dir / "study_notes.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return [out_path]
```

### 6.2 `trading.py`

```python
import csv
from pathlib import Path

_DISCLAIMER = (
    "> **DISCLAIMER:** This summary is generated by AI for informational purposes only "
    "and is **not financial advice**. Always conduct your own research.\n\n"
)

_PROMPTS = Path(__file__).parent.parent / "llm" / "prompts" / "domains"


class TradingProfile:
    name = "trading"
    description = "Financial markets, trading signals, and technical analysis."

    def chunk_prompt_addendum(self) -> str:
        return (_PROMPTS / "trading.txt").read_text(encoding="utf-8")

    def global_prompt_addendum(self) -> str:
        return ""

    def extra_output_schema(self) -> dict | None:
        return {"tickers_mentioned": ..., "signals": ..., "indicators": ..., "risk_warnings": ...}

    def post_process(self, summary: dict, fused: dict, output_dir: Path) -> list[Path]:
        created = []
        extras = {**summary, **summary.get("domain_extras", {})}

        # Prepend disclaimer to summary.md
        md_path = output_dir / "summary.md"
        if md_path.exists():
            original = md_path.read_text(encoding="utf-8")
            md_path.write_text(_DISCLAIMER + original, encoding="utf-8")

        # Write trade_log.csv
        signals = extras.get("signals", [])
        if signals:
            csv_path = output_dir / "trade_log.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "ticker", "action", "rationale", "timeframe"])
                writer.writeheader()
                for sig in signals:
                    writer.writerow({
                        "timestamp": _fmt_time(sig.get("timestamp", 0)),
                        "ticker": sig.get("ticker", ""),
                        "action": sig.get("action", ""),
                        "rationale": sig.get("rationale", ""),
                        "timeframe": sig.get("timeframe", ""),
                    })
            created.append(csv_path)

        return created
```

### 6.3 `medical.py`

```python
from pathlib import Path

_DISCLAIMER = (
    "> **DISCLAIMER:** This is an AI-generated educational summary. "
    "It is **not** for diagnostic or treatment decisions. Always verify with primary sources.\n\n"
)

_PROMPTS = Path(__file__).parent.parent / "llm" / "prompts" / "domains"


class MedicalProfile:
    name = "medical"
    description = "Medical education, conference talks, clinical lectures."

    def chunk_prompt_addendum(self) -> str:
        return (_PROMPTS / "medical.txt").read_text(encoding="utf-8")

    def global_prompt_addendum(self) -> str:
        return ""

    def extra_output_schema(self) -> dict | None:
        return None

    def post_process(self, summary: dict, fused: dict, output_dir: Path) -> list[Path]:
        created = []
        extras = {**summary, **summary.get("domain_extras", {})}

        # Prepend disclaimer
        md_path = output_dir / "summary.md"
        if md_path.exists():
            original = md_path.read_text(encoding="utf-8")
            md_path.write_text(_DISCLAIMER + original, encoding="utf-8")

        # Write clinical_notes.md
        lines = [_DISCLAIMER, "# Clinical Notes\n"]

        conditions = extras.get("conditions_discussed", [])
        if conditions:
            lines.append("## Conditions Discussed\n")
            for c in conditions:
                ts = _fmt_time(c.get("timestamp", 0))
                lines.append(f"- [{ts}] {c.get('name', '')}")
            lines.append("")

        treatments = extras.get("treatments_mentioned", [])
        if treatments:
            lines.append("## Treatments Mentioned\n")
            for t in treatments:
                ts = _fmt_time(t.get("timestamp", 0))
                lines.append(f"- [{ts}] **{t.get('name', '')}** ({t.get('type', '')})")
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
```

### 6.4 `law.py`

```python
from pathlib import Path

_DISCLAIMER = (
    "> **DISCLAIMER:** Not legal advice. Generated by AI; verify all citations independently.\n\n"
)

_PROMPTS = Path(__file__).parent.parent / "llm" / "prompts" / "domains"


class LawProfile:
    name = "law"
    description = "Legal lectures, case discussions, statutory analysis."

    def chunk_prompt_addendum(self) -> str:
        return (_PROMPTS / "law.txt").read_text(encoding="utf-8")

    def global_prompt_addendum(self) -> str:
        return ""

    def extra_output_schema(self) -> dict | None:
        return None

    def post_process(self, summary: dict, fused: dict, output_dir: Path) -> list[Path]:
        created = []
        extras = {**summary, **summary.get("domain_extras", {})}

        # Prepend disclaimer
        md_path = output_dir / "summary.md"
        if md_path.exists():
            original = md_path.read_text(encoding="utf-8")
            md_path.write_text(_DISCLAIMER + original, encoding="utf-8")

        # Write case_brief.md (IRAC format)
        lines = [_DISCLAIMER, "# Case Brief\n"]

        cases = extras.get("cases_cited", [])
        if cases:
            lines.append("## Cases Cited\n")
            for c in cases:
                ts = _fmt_time(c.get("timestamp", 0))
                lines.append(f"- [{ts}] **{c.get('case_name', '')}** — {c.get('citation', '')} ({c.get('jurisdiction', '')}): {c.get('relevance', '')}")
            lines.append("")

        statutes = extras.get("statutes", [])
        if statutes:
            lines.append("## Statutes Referenced\n")
            for s in statutes:
                ts = _fmt_time(s.get("timestamp", 0))
                lines.append(f"- [{ts}] {s.get('name', '')} § {s.get('section', '')}")
            lines.append("")

        principles = extras.get("legal_principles", [])
        if principles:
            lines.append("## Legal Principles\n")
            for p in principles:
                ts = _fmt_time(p.get("timestamp", 0))
                lines.append(f"- [{ts}] **{p.get('doctrine', '')}**: {p.get('explanation', '')}")
            lines.append("")

        brief_path = output_dir / "case_brief.md"
        brief_path.write_text("\n".join(lines), encoding="utf-8")
        created.append(brief_path)
        return created
```

### 6.5 `tutorial_strategy.py` + `pseudocode.py`

This is the flagship feature. `tutorial_strategy.py` calls `pseudocode.py` to render the Python skeleton.

```python
# src/domain/tutorial_strategy.py
import json
from pathlib import Path
from src.domain.pseudocode import render_python_skeleton, is_programming_tutorial

_PROMPTS = Path(__file__).parent.parent / "llm" / "prompts" / "domains"

PROGRAMMING_KEYWORDS = {
    "python", "function", "code", "script", "command", "git", "docker",
    "terminal", "bash", "javascript", "typescript", "react", "api", "sql",
    "install", "pip", "npm", "yarn", "class", "import", "module", "debug",
}


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
        created = []
        extras = {**summary, **summary.get("domain_extras", {})}

        task = extras.get("task", "")
        prerequisites = extras.get("prerequisites", [])
        steps = extras.get("steps", [])
        decision_points = extras.get("decision_points", [])
        video_url = fused.get("source_url", "")

        # Always write strategy.json
        strategy_data = {
            "task": task,
            "prerequisites": prerequisites,
            "steps": steps,
            "decision_points": decision_points,
        }
        json_path = output_dir / "strategy.json"
        json_path.write_text(json.dumps(strategy_data, indent=2, ensure_ascii=False), encoding="utf-8")
        created.append(json_path)

        # Always write strategy.md
        md_path = _write_strategy_md(output_dir, task, prerequisites, steps, decision_points)
        created.append(md_path)

        # Conditionally write strategy.py for programming tutorials
        if steps and is_programming_tutorial(task, steps):
            py_path = output_dir / "strategy.py"
            code = render_python_skeleton(steps, task, video_url, decision_points)
            py_path.write_text(code, encoding="utf-8")
            created.append(py_path)

        return created


def _write_strategy_md(output_dir, task, prerequisites, steps, decision_points) -> Path:
    lines = [f"# Tutorial Strategy: {task}\n"]
    if prerequisites:
        lines.append("## Prerequisites\n")
        for p in prerequisites:
            lines.append(f"- {p}")
        lines.append("")
    if steps:
        lines.append("## Steps\n")
        for s in steps:
            ts = _fmt_time(s.get("timestamp", 0))
            action = s.get("action", "")
            outcome = s.get("expected_outcome", "")
            params = s.get("parameters", {})
            param_str = ", ".join(f"`{k}={v}`" for k, v in params.items()) if params else ""
            lines.append(f"### Step {s.get('step_number', '?')}: {action} [{ts}]")
            if param_str:
                lines.append(f"- Parameters: {param_str}")
            if outcome:
                lines.append(f"- Expected: {outcome}")
            lines.append("")
    if decision_points:
        lines.append("## Decision Points\n")
        for dp in decision_points:
            lines.append(f"- **Step {dp.get('step_number', '?')}**: If `{dp.get('condition', '')}` → {dp.get('branch_yes', '')} | else → {dp.get('branch_no', '')}")
        lines.append("")
    md_path = output_dir / "strategy.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path
```

```python
# src/domain/pseudocode.py
"""Generate a Python skeleton from extracted tutorial steps."""
from __future__ import annotations
import ast
import re

PROGRAMMING_KEYWORDS = {
    "python", "function", "code", "script", "command", "git", "docker",
    "terminal", "bash", "javascript", "typescript", "react", "api", "sql",
    "install", "pip", "npm", "yarn", "class", "import", "module", "debug",
}


def is_programming_tutorial(task: str, steps: list[dict]) -> bool:
    text = (task + " ".join(s.get("action", "") for s in steps)).lower()
    return any(kw in text for kw in PROGRAMMING_KEYWORDS)


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:40]


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def render_python_skeleton(
    steps: list[dict],
    task: str,
    source_url: str = "",
    decision_points: list[dict] = None,
) -> str:
    """
    Render a Python skeleton file with one function per step.

    Validates the output with ast.parse; falls back to comment-only block on failure.
    """
    decision_points = decision_points or []
    dp_map: dict[int, dict] = {dp["step_number"]: dp for dp in decision_points}

    header_lines = [
        f'"""',
        f"Auto-generated workflow skeleton.",
        f"Task: {task}",
    ]
    if source_url:
        header_lines.append(f"Source: {source_url}")
    header_lines += ['"""', "", ""]

    func_lines: list[str] = []
    call_names: list[str] = []

    for s in steps:
        num = s.get("step_number", 0)
        action = s.get("action", "step")
        ts = _fmt_time(float(s.get("timestamp", 0)))
        outcome = s.get("expected_outcome", "")
        params = s.get("parameters", {})
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

        body = "    # TODO: implement\n    ..."

        # Add decision branch as inline comment if present
        if num in dp_map:
            dp = dp_map[num]
            body += (
                f"\n    # Decision: if {dp['condition']}:\n"
                f"    #     -> {dp['branch_yes']}\n"
                f"    # else:\n"
                f"    #     -> {dp['branch_no']}"
            )

        func_lines.append(f"def {func_name}():")
        func_lines.extend(doc_lines)
        func_lines.append(body)
        func_lines.append("")
        func_lines.append("")

    main_body = ["if __name__ == '__main__':"]
    for name in call_names:
        main_body.append(f"    {name}()")

    code = "\n".join(header_lines + func_lines + main_body + [""])

    # Validate — fall back to single commented block on parse error
    try:
        ast.parse(code)
    except SyntaxError:
        fallback = ["# Auto-generated workflow (parse validation failed)\n"]
        for s in steps:
            ts = _fmt_time(float(s.get("timestamp", 0)))
            fallback.append(f"# Step {s.get('step_number', '?')} [{ts}]: {s.get('action', '')}")
        code = "\n".join(fallback)

    return code
```

---

## 7. Helper shared by all profiles

Add to `src/domain/base.py`:

```python
def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
```

Import and use this in each profile file.

---

## 8. Formatter Wiring

In `src/output/formatter.py`, locate the `format_outputs()` function and add after all existing output writes:

```python
# Domain post-processing
if domain:
    try:
        from src.domain.registry import get_domain
        profile = get_domain(domain)

        # Load summary dict (already written as report.json)
        report_json_path = output_dir / "report.json"
        summary_dict: dict = json.loads(report_json_path.read_text(encoding="utf-8")) if report_json_path.exists() else {}

        # Load fused dict
        fused_json_path = paths.fused_json  # or pass fused_path explicitly
        fused_dict: dict = json.loads(fused_json_path.read_text(encoding="utf-8")) if fused_json_path.exists() else {}

        extra_paths = profile.post_process(summary_dict, fused_dict, output_dir)
        LOGGER.info(f"Domain '{domain}' generated {len(extra_paths)} extra file(s)")
        result.extra_files = extra_paths
    except Exception as exc:
        LOGGER.warning(f"Domain post-processing failed (non-fatal): {exc}")
```

---

## 9. Pipeline Wiring

In `src/stages/format.py`, ensure `domain` is read from `StageCtx` and forwarded to `format_outputs()`. In `src/stages/summarize.py`, ensure `domain` is read from `StageCtx` and forwarded to `summarize()`. Both already have `domain` in `StageCtx` — verify they pass it through.

---

## 10. Tests

### `tests/domain/conftest.py`
```python
import pytest

CANNED_FUSED = {
    "run_id": "test-run",
    "events": [],
    "duration_sec": 300.0,
    "source_url": "https://example.com/video",
}

CANNED_SUMMARY = {
    "full_summary": "A tutorial about Python.",
    "key_points": [{"timestamp": 10.0, "text": "Install Python", "confidence": "high"}],
    "chapters": [],
    "events": [],
    "learning_objectives": ["Understand variables"],
    "worked_examples": [],
    "definitions": [{"term": "Variable", "definition": "Named storage", "timestamp": 20.0}],
    "tickers_mentioned": ["AAPL"],
    "signals": [{"timestamp": 30.0, "ticker": "AAPL", "action": "buy", "rationale": "Breakout", "timeframe": "weekly"}],
    "conditions_discussed": [{"name": "Hypertension", "timestamp": 15.0}],
    "treatments_mentioned": [{"name": "Lisinopril", "type": "drug", "timestamp": 25.0}],
    "key_clinical_points": ["Monitor BP regularly."],
    "cases_cited": [{"case_name": "Roe v. Wade", "citation": "410 U.S. 113", "jurisdiction": "SCOTUS", "timestamp": 40.0, "relevance": "precedent"}],
    "statutes": [],
    "legal_principles": [],
    "task": "Build a Python web scraper",
    "prerequisites": ["Python 3.11", "pip"],
    "steps": [
        {"step_number": 1, "timestamp": 5.0, "action": "Install requests library", "parameters": {}, "expected_outcome": "requests installed", "screenshot_frame_index": None},
        {"step_number": 2, "timestamp": 60.0, "action": "Write scraping function", "parameters": {"url": "target-url"}, "expected_outcome": "HTML returned", "screenshot_frame_index": None},
    ],
    "decision_points": [],
}

@pytest.fixture
def tmp_output(tmp_path):
    (tmp_path / "summary.md").write_text("# Summary\n\nTest content.\n", encoding="utf-8")
    return tmp_path

@pytest.fixture
def canned_summary():
    return dict(CANNED_SUMMARY)

@pytest.fixture
def canned_fused():
    return dict(CANNED_FUSED)
```

### Key test cases (in respective test files):

1. **Registry** (`test_registry.py`): all 5 names resolve; unknown name raises `UnknownDomainError`.
2. **Education** (`test_education.py`): `post_process` creates `study_notes.md` containing "Learning Objectives" and "Key Definitions".
3. **Trading** (`test_trading.py`): `post_process` creates `trade_log.csv` with header row; disclaimer prepended to `summary.md`.
4. **Medical** (`test_medical.py`): `post_process` creates `clinical_notes.md`; disclaimer prepended to `summary.md`.
5. **Law** (`test_law.py`): `post_process` creates `case_brief.md`; disclaimer prepended to `summary.md`.
6. **Tutorial-strategy** (`test_tutorial_strategy.py`):
   - `strategy.json` written with correct keys.
   - `strategy.md` written with "Steps" heading.
   - `strategy.py` written and parses cleanly (`ast.parse`).
   - Non-programming tutorial (task has no code keywords) → no `strategy.py`.
7. **Addendum injection** (`test_registry.py`): call `get_chunk_prompt(events_json, domain_addendum=profile.chunk_prompt_addendum())` and assert addendum text is in the returned user message.
8. **No-domain regression**: verify `DOMAINS["education"].post_process(...)` doesn't run when domain is `None`.

---

## 11. Phased Execution

| Phase | Task | Effort |
|-------|------|--------|
| A | Create `src/domain/` skeleton (base.py, registry.py, __init__.py) | 30 min |
| B | Write 5 prompt `.txt` files | 20 min |
| C | Implement education.py, trading.py, medical.py, law.py | 1.5 hr |
| D | Implement tutorial_strategy.py + pseudocode.py | 2 hr |
| E | Patch prompts_system.py to accept domain_addendum | 20 min |
| F | Patch summarizer.py to resolve and pass addenda | 30 min |
| G | Patch formatter.py to call post_process | 30 min |
| H | Verify stages/summarize.py and stages/format.py pass domain through | 15 min |
| I | Write all tests | 1.5 hr |
| J | Run pytest, fix issues | 30 min |

**Total estimated effort: ~7 hours**

---

## 12. Acceptance Criteria

- [ ] `python -m src.pipeline --url <URL> --domain education` produces `study_notes.md` in `data/output/<run_id>/`.
- [ ] `python -m src.pipeline --url <URL> --domain trading` produces `trade_log.csv` with disclaimer in `summary.md`.
- [ ] `python -m src.pipeline --url <URL> --domain tutorial-strategy` produces `strategy.json`, `strategy.md`, and (for programming tutorials) `strategy.py` that passes `ast.parse`.
- [ ] `python -m src.pipeline --url <URL>` (no domain) produces identical output to pre-domain baseline.
- [ ] All unit tests pass without API keys (domain post-processing uses only `summary.json` already on disk).
- [ ] Unknown domain name raises `UnknownDomainError` with helpful message listing valid names.

---

## 13. Definition of Done

A developer runs `python -m src.pipeline --url <python-tutorial-url> --domain tutorial-strategy` and finds in `data/output/<run_id>/`: the standard `summary.md`, `report.json`, `chapters.txt` **plus** `strategy.json`, `strategy.md`, and `strategy.py` — where `strategy.py` imports cleanly, has one function per tutorial step, and each function's docstring contains the timestamp and expected outcome from the video.
