from __future__ import annotations

import pytest

from tests.helpers import REPO_ROOT
from tradesense.config import Settings, load_settings


@pytest.fixture
def settings() -> Settings:
    """The repository's real config files (no .env loading: tests stay hermetic)."""
    return load_settings(REPO_ROOT, load_env=False)
