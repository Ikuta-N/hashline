import hashline


def test_package_exposes_version() -> None:
    assert hashline.__version__
