import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "network: mark test as requiring live network access (skip with -m 'not network')",
    )
