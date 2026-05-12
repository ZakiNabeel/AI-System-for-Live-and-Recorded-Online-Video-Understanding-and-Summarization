"""Tests for the education domain profile."""

from src.domain.education import EducationProfile


def test_study_notes_created(tmp_output, canned_summary, canned_fused):
    profile = EducationProfile()
    created = profile.post_process(canned_summary, canned_fused, tmp_output)
    paths = [p.name for p in created]
    assert "study_notes.md" in paths


def test_study_notes_has_objectives(tmp_output, canned_summary, canned_fused):
    profile = EducationProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    content = (tmp_output / "study_notes.md").read_text(encoding="utf-8")
    assert "Learning Objectives" in content
    assert "Understand HTTP" in content


def test_study_notes_has_definitions(tmp_output, canned_summary, canned_fused):
    profile = EducationProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    content = (tmp_output / "study_notes.md").read_text(encoding="utf-8")
    assert "Key Definitions" in content
    assert "Scraping" in content


def test_study_notes_has_worked_examples(tmp_output, canned_summary, canned_fused):
    profile = EducationProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    content = (tmp_output / "study_notes.md").read_text(encoding="utf-8")
    assert "Worked Examples" in content
    assert "Fetch a web page" in content


def test_chunk_addendum_non_empty():
    profile = EducationProfile()
    addendum = profile.chunk_prompt_addendum()
    assert "DOMAIN MODE: EDUCATION" in addendum
    assert "learning_objectives" in addendum
