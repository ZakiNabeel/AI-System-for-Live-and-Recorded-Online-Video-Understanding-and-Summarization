"""Domain profile registry."""

from __future__ import annotations

from .base import DomainProfile
from .education import EducationProfile
from .law import LawProfile
from .medical import MedicalProfile
from .trading import TradingProfile
from .tutorial_strategy import TutorialStrategyProfile

DOMAINS: dict[str, DomainProfile] = {
    "education": EducationProfile(),
    "trading": TradingProfile(),
    "medical": MedicalProfile(),
    "law": LawProfile(),
    "tutorial-strategy": TutorialStrategyProfile(),
}


class UnknownDomainError(ValueError):
    def __init__(self, name: str) -> None:
        super().__init__(
            f"Unknown domain '{name}'. Valid domains: {sorted(DOMAINS)}"
        )


def get_domain(name: str) -> DomainProfile:
    """Return the profile for *name*, raising UnknownDomainError if not found."""
    if name not in DOMAINS:
        raise UnknownDomainError(name)
    return DOMAINS[name]
