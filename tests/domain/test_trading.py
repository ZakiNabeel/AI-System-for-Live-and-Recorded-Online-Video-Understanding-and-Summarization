"""Tests for the trading domain profile."""

import csv
from src.domain.trading import TradingProfile, _DISCLAIMER


def test_trade_log_csv_created(tmp_output, canned_summary, canned_fused):
    profile = TradingProfile()
    created = profile.post_process(canned_summary, canned_fused, tmp_output)
    paths = [p.name for p in created]
    assert "trade_log.csv" in paths


def test_trade_log_has_correct_headers(tmp_output, canned_summary, canned_fused):
    profile = TradingProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    with (tmp_output / "trade_log.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames) == {"timestamp", "ticker", "action", "rationale", "timeframe"}


def test_trade_log_has_signal_row(tmp_output, canned_summary, canned_fused):
    profile = TradingProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    with (tmp_output / "trade_log.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["action"] == "buy"


def test_disclaimer_prepended_to_summary_md(tmp_output, canned_summary, canned_fused):
    profile = TradingProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    content = (tmp_output / "summary.md").read_text(encoding="utf-8")
    assert content.startswith(">")
    assert "not financial advice" in content


def test_no_signals_no_csv(tmp_output, canned_fused):
    profile = TradingProfile()
    summary_no_signals = {"full_summary": "test", "key_points": [], "signals": []}
    created = profile.post_process(summary_no_signals, canned_fused, tmp_output)
    assert not any(p.name == "trade_log.csv" for p in created)
