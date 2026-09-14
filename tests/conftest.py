"""Test environment.

The committed .env targets the container network (postgres:5432, host.docker.internal).
Tests run on the host, so they need the host-facing equivalents. Environment variables
take precedence over .env in pydantic-settings, and setdefault means a real exported
value still wins — so CI can point these anywhere.
"""

import os

os.environ.setdefault("DATABASE_URL", "postgresql://policyiq:policyiq@localhost:5432/policyiq")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
