"""Test environment.

The committed .env targets the container network (postgres:5432, host.docker.internal).
Tests run on the host, so they need the host-facing equivalents. Environment variables
take precedence over .env in pydantic-settings, and setdefault means a real exported
value still wins — so CI can point these anywhere.
"""

import os
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://policyiq:policyiq@localhost:5432/policyiq")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")


import pytest  # noqa: E402

from tests.fixtures import write_synthetic_policy  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_policy(tmp_path_factory) -> Path:
    """A generated policy PDF. Always available, including in CI."""
    return write_synthetic_policy(tmp_path_factory.mktemp("corpus") / "specimen-policy.pdf")


@pytest.fixture(scope="session")
def sample_pdf(synthetic_policy) -> Path:
    """Prefer a real corpus document when one is present, else the synthetic one.

    Local runs exercise the real documents; CI exercises the synthetic one. Both
    paths stay tested.
    """
    real = sorted(Path("data/policies").glob("*.pdf"))
    return real[0] if real else synthetic_policy
