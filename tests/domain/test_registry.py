"""Tests for the domain registry."""

import pytest
from src.domain.registry import DOMAINS, UnknownDomainError, get_domain


def test_all_five_profiles_registered():
    expected = {"education", "trading", "medical", "law", "tutorial-strategy"}
    assert set(DOMAINS.keys()) == expected


def test_get_domain_returns_profile():
    profile = get_domain("education")
    assert profile.name == "education"


def test_get_domain_unknown_raises():
    with pytest.raises(UnknownDomainError) as exc_info:
        get_domain("astrology")
    assert "astrology" in str(exc_info.value)
    assert "education" in str(exc_info.value)


def test_all_profiles_have_chunk_addendum():
    for name, profile in DOMAINS.items():
        addendum = profile.chunk_prompt_addendum()
        assert isinstance(addendum, str), f"{name}: chunk_prompt_addendum() must return str"
        assert len(addendum) > 10, f"{name}: chunk addendum is too short"


def test_all_profiles_implement_protocol():
    from src.domain.base import DomainProfile
    for name, profile in DOMAINS.items():
        assert isinstance(profile, DomainProfile), f"{name} does not implement DomainProfile"
