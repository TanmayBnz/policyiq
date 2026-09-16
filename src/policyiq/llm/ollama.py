import httpx

from policyiq.config import settings


class OllamaProvider:
    """Generation against an Ollama server.

    Ollama runs on the host rather than in a container: passing a GPU into Docker needs
    the container toolkit and correct runtime configuration, which on a hybrid-graphics
    laptop is a reliable way to lose an afternoon. The container reaches it through the
    Docker host gateway instead, which costs one line of Compose config. That is a
    trade-off rather than a limitation, and this class is the reason it stays one -
    relocating the server later is a change of `base_url`.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self._timeout = timeout
        # Injectable so tests can drive a real httpx client over a stub transport,
        # exercising request construction and parsing without a model server present.
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def generate(self, prompt: str) -> str:
        """Ask the model to answer, and fail loudly if it cannot.

        Raising rather than returning an empty string is deliberate: an empty answer
        would be rendered to a user as the policy saying nothing on the subject, which
        is a wrong answer wearing the clothes of a correct one.
        """
        try:
            response = self._client.post(
                "/api/generate",
                # Ollama streams by default, emitting one JSON object per token as
                # newline-delimited JSON. Parsing that as a single object fails, so
                # streaming is turned off explicitly rather than by omission.
                #
                # Temperature 0 makes the model pick its most likely next word every
                # time instead of sampling. Without it the same question over the same
                # passages was answered on one run and refused on the next, which makes
                # an answer impossible to test or evaluate. Determinism is not
                # correctness: a wrong answer now stays reliably wrong.
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"{self.model} at {self.base_url} returned {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"could not reach {self.base_url}: {exc}") from exc

        return response.json()["response"].strip()

    def healthy(self) -> bool:
        """Whether the server is reachable.

        Never raises. Readiness calls this on every probe, and an exception here would
        turn an orderly 503 into a 500 from the readiness endpoint itself - reporting
        the service as broken when the truth is that a dependency is unavailable.

        This checks reachability, not that the configured model is loaded. A pull that
        has not happened would pass here and fail at generation.
        """
        try:
            return self._client.get("/api/tags", timeout=self._timeout).status_code == 200
        except httpx.HTTPError:
            return False


def build_from_settings() -> OllamaProvider:
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout_seconds,
    )
