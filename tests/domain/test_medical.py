"""Tests for the medical domain profile."""

from src.domain.medical import MedicalProfile


def test_clinical_notes_created(tmp_output, canned_summary, canned_fused):
    profile = MedicalProfile()
    created = profile.post_process(canned_summary, canned_fused, tmp_output)
    assert any(p.name == "clinical_notes.md" for p in created)


def test_clinical_notes_has_conditions(tmp_output, canned_summary, canned_fused):
    profile = MedicalProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    content = (tmp_output / "clinical_notes.md").read_text(encoding="utf-8")
    assert "Hypertension" in content


def test_clinical_notes_has_treatments(tmp_output, canned_summary, canned_fused):
    profile = MedicalProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    content = (tmp_output / "clinical_notes.md").read_text(encoding="utf-8")
    assert "Lisinopril" in content


def test_disclaimer_prepended(tmp_output, canned_summary, canned_fused):
    profile = MedicalProfile()
    profile.post_process(canned_summary, canned_fused, tmp_output)
    content = (tmp_output / "summary.md").read_text(encoding="utf-8")
    assert "not" in content.lower()
    assert "diagnostic" in content.lower()
