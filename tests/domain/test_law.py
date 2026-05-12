"""Tests for the law domain profile."""

from src.domain.law import LawProfile


def test_case_brief_created(tmp_output, canned_summary, canned_fused):
    profile = LawProfile()
    created = profile.post_process(canned_summary, canned_fused, tmp_output)
    assert any(p.name == "case_brief.md" for p in created)


def test_case_brief_has_case(tmp_output, canned_summary, canned_fused):
    profile = LawProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    content = (tmp_output / "case_brief.md").read_text(encoding="utf-8")
    assert "Brown v. Board" in content
    assert "347 U.S. 483" in content


def test_disclaimer_prepended(tmp_output, canned_summary, canned_fused):
    profile = LawProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    content = (tmp_output / "summary.md").read_text(encoding="utf-8")
    assert "Not legal advice" in content
