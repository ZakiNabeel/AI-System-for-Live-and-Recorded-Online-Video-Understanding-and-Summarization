"""Tests for the tutorial-strategy domain profile."""

import ast
import json

from src.domain.tutorial_strategy import TutorialStrategyProfile
from src.domain.pseudocode import is_programming_tutorial, render_python_skeleton


def test_strategy_json_created(tmp_output, canned_summary, canned_fused):
    profile = TutorialStrategyProfile()
    created = profile.post_process(canned_summary, canned_fused, tmp_output)
    assert any(p.name == "strategy.json" for p in created)


def test_strategy_md_created(tmp_output, canned_summary, canned_fused):
    profile = TutorialStrategyProfile()
    created = profile.post_process(canned_summary, canned_fused, tmp_output)
    assert any(p.name == "strategy.md" for p in created)


def test_strategy_json_has_required_keys(tmp_output, canned_summary, canned_fused):
    profile = TutorialStrategyProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    data = json.loads((tmp_output / "strategy.json").read_text(encoding="utf-8"))
    for key in ("task", "prerequisites", "steps", "decision_points"):
        assert key in data, f"Missing key: {key}"


def test_strategy_md_has_steps_heading(tmp_output, canned_summary, canned_fused):
    profile = TutorialStrategyProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    content = (tmp_output / "strategy.md").read_text(encoding="utf-8")
    assert "## Steps" in content
    assert "Install requests library" in content


def test_strategy_py_created_for_programming_tutorial(tmp_output, canned_summary, canned_fused):
    # task contains "Python" and steps mention "scraping function"
    profile = TutorialStrategyProfile()
    created = profile.post_process(canned_summary, canned_fused, tmp_output)
    assert any(p.name == "strategy.py" for p in created)


def test_strategy_py_parses_cleanly(tmp_output, canned_summary, canned_fused):
    profile = TutorialStrategyProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    code = (tmp_output / "strategy.py").read_text(encoding="utf-8")
    # Should not raise
    ast.parse(code)


def test_strategy_py_has_correct_function_count(tmp_output, canned_summary, canned_fused):
    profile = TutorialStrategyProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    code = (tmp_output / "strategy.py").read_text(encoding="utf-8")
    tree = ast.parse(code)
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert len(funcs) == len(canned_summary["steps"])


def test_no_strategy_py_for_non_programming(tmp_output, canned_fused):
    profile = TutorialStrategyProfile()
    non_prog_summary = {
        "task": "Bake a sourdough loaf",
        "prerequisites": ["flour", "water"],
        "steps": [
            {"step_number": 1, "timestamp": 0.0, "action": "Mix flour and water", "parameters": {}, "expected_outcome": "Dough formed", "screenshot_frame_index": None},
        ],
        "decision_points": [],
    }
    created = profile.post_process(non_prog_summary, canned_fused, tmp_output)
    assert not any(p.name == "strategy.py" for p in created)


def test_is_programming_tutorial_detects_python():
    steps = [{"action": "run pip install"}]
    assert is_programming_tutorial("Python tutorial", steps) is True


def test_is_programming_tutorial_rejects_cooking():
    steps = [{"action": "mix ingredients"}]
    assert is_programming_tutorial("How to bake bread", steps) is False


def test_render_skeleton_parses():
    steps = [
        {"step_number": 1, "timestamp": 0, "action": "Create file", "parameters": {}, "expected_outcome": "File created"},
    ]
    code = render_python_skeleton(steps, "File creation tutorial")
    ast.parse(code)
