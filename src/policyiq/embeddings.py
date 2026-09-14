from functools import lru_cache

from fastembed import TextEmbedding

from policyiq.config import settings


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    """Loaded once per process. The first call downloads the model (~130MB) and caches
    it on disk.

    ONNX Runtime rather than PyTorch: the same bge-small weights, roughly 100MB of
    dependencies instead of 2.5GB. PyTorch is a training stack, and at inference time
    on CPU it buys nothing here while dominating the container image size.
    """
    return TextEmbedding(model_name=settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed documents for storage."""
    if not texts:
        return []
    return [vector.tolist() for vector in _model().embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embed a search query.

    Kept separate from embed_texts because asymmetric retrieval models prefix queries
    and documents differently. Splitting them now means swapping in such a model later
    touches this file only.
    """
    return [vector.tolist() for vector in _model().query_embed([text])][0]
