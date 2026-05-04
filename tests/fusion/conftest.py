"""Shared pytest configuration for fusion tests."""

from __future__ import annotations

import logging
import pytest


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
