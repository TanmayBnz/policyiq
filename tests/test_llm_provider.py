"""The model layer.

Split deliberately into two kinds of test. The unit tests drive a real httpx client
over a stub transport, so request construction, response parsing and failure handling
are exercised everywhere - including CI, where no model server exists. The integration
tests at the bottom talk to a real Ollama and are the only ones that skip.

Mocking the transport rather than our own code keeps the assertions on behaviour: the
request that would go over the wire is a real request object, not a recorded call.
"""

import httpx
import pytest

from policyiq.config import settings
from policyiq.llm import get_provider
from policyiq.llm.base import LLMProvider
from policyiq.llm.ollama import OllamaProvider


def provider_over(handler, **kwargs) -> OllamaProvider:
    """An OllamaProvider whose HTTP calls are answered by `handler`."""
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://stub:11434")
    return OllamaProvider(base_url="http://stub:11434", model="test-model", client=client, **kwargs)


def test_the_ollama_provider_satisfies_the_protocol():
    """Structural, not inheritance-based. A provider for a different vendor needs to
    satisfy the same shape without importing anything from this one."""
    assert isinstance(provider_over(lambda r: httpx.Response(200, json={})), LLMProvider)


def test_generate_sends_the_model_and_prompt_and_returns_the_answer():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        seen["path"] = request.url.path
        return httpx.Response(200, json={"response": "  36 months.  "})

    answer = provider_over(handler).generate("How long is the waiting period?")

    assert seen["path"] == "/api/generate"
    assert seen["model"] == "test-model"
    assert seen["prompt"] == "How long is the waiting period?"
    assert answer == "36 months.", "surrounding whitespace should not reach the caller"


def test_generate_asks_for_a_single_response_rather_than_a_stream():
    """Ollama streams by default, returning one JSON object per token as NDJSON. Parsing
    that as a single object fails, so streaming has to be turned off explicitly."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"response": "ok"})

    provider_over(handler).generate("anything")

    assert seen["stream"] is False


def test_generate_raises_when_the_model_server_returns_an_error():
    """A failure here must not be mistaken for an empty answer. An answer that silently
    becomes '' would be presented to a user as the policy saying nothing."""
    provider = provider_over(lambda r: httpx.Response(500, text="model runner crashed"))

    with pytest.raises(RuntimeError, match="500"):
        provider.generate("anything")


def test_generate_raises_when_the_server_is_unreachable():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(RuntimeError):
        provider_over(refuse).generate("anything")


def test_healthy_is_true_when_the_server_answers():
    assert provider_over(lambda r: httpx.Response(200, json={"models": []})).healthy() is True


def test_healthy_is_false_rather_than_raising_when_the_server_is_down():
    """Readiness calls this on every probe. If it raised, a model server outage would
    turn an orderly 503 into a 500 from the readiness endpoint itself."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    assert provider_over(refuse).healthy() is False


def test_healthy_is_false_when_the_server_answers_with_an_error():
    assert provider_over(lambda r: httpx.Response(503)).healthy() is False


def test_get_provider_is_configured_from_settings():
    provider = get_provider()

    assert isinstance(provider, LLMProvider)
    assert provider.model == settings.ollama_model
    assert provider.base_url == settings.ollama_base_url


# --- integration: a real model server ---------------------------------------------
#
# These are the only tests here that skip. The unit tests above cover request shape,
# parsing and failure handling, so a CI run without Ollama still proves the logic
# rather than passing vacuously.


@pytest.fixture(scope="module")
def live_provider() -> LLMProvider:
    provider = get_provider()
    if not provider.healthy():
        pytest.skip(f"no model server at {settings.ollama_base_url}")
    return provider


def test_a_real_model_returns_non_empty_text(live_provider: LLMProvider):
    answer = live_provider.generate("Reply with exactly one word: ready")

    assert isinstance(answer, str)
    assert answer.strip()


def test_a_real_model_follows_a_simple_instruction(live_provider: LLMProvider):
    """Proves the model is being prompted rather than merely reached. A server that
    echoed or returned a canned string would pass the test above and fail this one."""
    answer = live_provider.generate(
        "Answer with a single digit and nothing else. What is two plus three?"
    )

    assert "5" in answer
