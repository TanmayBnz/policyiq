"""The model layer.

Selection lives here so that nothing above this package names a vendor. Today there is
one implementation; adding a hosted provider means a new module and a branch in
`get_provider`, and no change to retrieval or answering.
"""

from functools import lru_cache

from policyiq.llm.base import LLMProvider
from policyiq.llm.ollama import OllamaProvider, build_from_settings

__all__ = ["LLMProvider", "OllamaProvider", "get_provider"]


@lru_cache(maxsize=1)
def get_provider() -> LLMProvider:
    """One provider per process, created lazily.

    Cached because it owns an HTTP client with a connection pool. Building a new one
    per request would open a fresh connection every time, and readiness probes call
    this on a timer.
    """
    return build_from_settings()
