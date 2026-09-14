import math

from policyiq.embeddings import embed_query, embed_texts


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def test_embeddings_have_expected_dimension():
    vectors = embed_texts(["hospitalisation expenses are covered"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384


def test_batch_size_is_preserved():
    assert len(embed_texts(["alpha", "beta", "gamma"])) == 3


def test_related_text_scores_higher_than_unrelated():
    """The only test here that proves the embeddings carry meaning rather than
    just being correctly-shaped arrays of numbers."""
    query = embed_query("what is covered for hospital admission")
    related = embed_texts(["The policy covers inpatient hospitalisation expenses."])[0]
    unrelated = embed_texts(["The quick brown fox jumps over the lazy dog."])[0]
    assert _cosine(query, related) > _cosine(query, unrelated)


def test_empty_batch_returns_empty_list():
    assert embed_texts([]) == []
