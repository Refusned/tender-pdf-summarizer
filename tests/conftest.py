from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    return (FIXTURES / "sample_tender.pdf").read_bytes()
