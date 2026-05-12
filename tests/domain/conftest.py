"""Shared fixtures for domain tests."""

import pytest


CANNED_FUSED = {
    "run_id": "test-run",
    "events": [],
    "duration_sec": 300.0,
    "source_url": "https://example.com/video",
}

CANNED_SUMMARY = {
    "run_id": "test-run",
    "full_summary": "A tutorial about Python web scraping.",
    "short_summary": "Python web scraping tutorial.",
    "key_points": [{"timestamp": 10.0, "text": "Install requests", "confidence": "high"}],
    "chapters": [],
    "events": [],
    # Education extras
    "learning_objectives": ["Understand HTTP requests", "Parse HTML with BeautifulSoup"],
    "worked_examples": [
        {
            "timestamp": 60.0,
            "problem_statement": "Fetch a web page",
            "key_steps": ["Import requests", "Call requests.get(url)"],
            "final_answer": "Response object with HTML content",
        }
    ],
    "definitions": [
        {"term": "Scraping", "definition": "Extracting data from websites", "timestamp": 15.0}
    ],
    # Trading extras
    "tickers_mentioned": ["AAPL"],
    "signals": [
        {
            "timestamp": 30.0,
            "ticker": "AAPL",
            "action": "buy",
            "rationale": "Breakout above resistance",
            "timeframe": "weekly",
        }
    ],
    "indicators": [],
    "risk_warnings": ["Past performance is not indicative of future results."],
    # Medical extras
    "conditions_discussed": [{"name": "Hypertension", "timestamp": 15.0}],
    "treatments_mentioned": [{"name": "Lisinopril", "type": "drug", "timestamp": 25.0}],
    "evidence_levels": [],
    "key_clinical_points": ["Monitor blood pressure regularly."],
    # Law extras
    "cases_cited": [
        {
            "case_name": "Brown v. Board of Education",
            "citation": "347 U.S. 483",
            "jurisdiction": "SCOTUS",
            "timestamp": 40.0,
            "relevance": "Landmark desegregation case",
        }
    ],
    "statutes": [],
    "legal_principles": [],
    "arguments": [],
    # Tutorial-strategy extras
    "task": "Build a Python web scraper",
    "prerequisites": ["Python 3.11", "pip"],
    "steps": [
        {
            "step_number": 1,
            "timestamp": 5.0,
            "action": "Install requests library",
            "parameters": {},
            "expected_outcome": "requests installed successfully",
            "screenshot_frame_index": None,
        },
        {
            "step_number": 2,
            "timestamp": 60.0,
            "action": "Write scraping function",
            "parameters": {"url": "target-url"},
            "expected_outcome": "HTML content returned",
            "screenshot_frame_index": None,
        },
    ],
    "decision_points": [],
}


@pytest.fixture
def tmp_output(tmp_path):
    """Temp output dir with a pre-existing summary.md."""
    (tmp_path / "summary.md").write_text("# Summary\n\nTest content.\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def canned_summary():
    return dict(CANNED_SUMMARY)


@pytest.fixture
def canned_fused():
    return dict(CANNED_FUSED)
