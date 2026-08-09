from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def notes_dir() -> Path:
    """The synthetic import corpus. Tests never read notes from anywhere else."""
    return FIXTURES / "notes"
