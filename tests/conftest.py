from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_background_indexing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the web app's startup indexer out of every test.

    Entering a TestClient runs the lifespan, which starts a thread that embeds
    anything unembedded. On a machine with the `ml` extra installed that means
    loading a model and writing to the same database the test is using -- slow,
    and racy enough to fail unrelated tests. CI has no extra installed and so
    never saw it, which is exactly why it has to be switched off here rather
    than left to luck.

    The tests for the gate itself opt back out with their own `delenv`, which
    runs after this fixture.
    """
    monkeypatch.setenv("HASHLINE_NO_INDEX", "1")
    # Models are cached for the life of the process, so a fake one swapped in
    # by a test would otherwise be handed to every test after it.
    from hashline.ml.hybrid import forget_models

    forget_models()


@pytest.fixture
def notes_dir() -> Path:
    """The synthetic import corpus. Tests never read notes from anywhere else."""
    return FIXTURES / "notes"
