"""Reciprocal rank fusion.

A pure function over lists, so no database: these run anywhere, and each ranking is
built by hand to state exactly what it tests.
"""

import pytest

from policyiq.retrieval.fusion import reciprocal_rank_fusion
from policyiq.retrieval.vector import RetrievedChunk


def chunk(chunk_id: int, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, document_id=1, filename="a.pdf", page_number=1,
        chunk_index=chunk_id, content=f"chunk {chunk_id}", score=score,
    )


def ids(chunks: list[RetrievedChunk]) -> list[int]:
    return [c.chunk_id for c in chunks]


def test_agreement_beats_a_single_first_place():
    """The reason to fuse at all. Chunk 7 is 3rd for both retrievers; chunk 1 is 1st for
    one and missing from the other. 2/63 is more than 1/61, so agreement wins. If this
    fails, fusion is just "whichever list is longer" or "the first list"."""
    vector = [chunk(1), chunk(2), chunk(7)]
    keyword = [chunk(8), chunk(9), chunk(7)]

    assert ids(reciprocal_rank_fusion([vector, keyword], top_k=1)) == [7]


def test_a_chunk_found_by_both_appears_once():
    fused = reciprocal_rank_fusion([[chunk(1), chunk(2)], [chunk(2), chunk(1)]], top_k=10)

    assert sorted(ids(fused)) == [1, 2]


def test_scores_are_the_sum_of_reciprocal_ranks():
    fused = reciprocal_rank_fusion([[chunk(1), chunk(2)], [chunk(2)]], top_k=10, k=60)
    by_id = {c.chunk_id: c.score for c in fused}

    assert by_id[2] == pytest.approx(1 / 62 + 1 / 61)
    assert by_id[1] == pytest.approx(1 / 61)


def test_the_retrievers_own_scores_are_ignored():
    """Only positions count. Chunk 1 has an enormous score in the second list; if scores
    leaked into fusion it would win."""
    fused = reciprocal_rank_fusion(
        [[chunk(2, 0.9), chunk(1, 0.8)], [chunk(2, 0.01), chunk(1, 500.0)]], top_k=2
    )

    assert ids(fused) == [2, 1]


def test_results_are_best_first_and_cut_to_top_k():
    rankings = [[chunk(i) for i in range(1, 11)], [chunk(i) for i in range(10, 0, -1)]]

    fused = reciprocal_rank_fusion(rankings, top_k=4)

    assert len(fused) == 4
    scores = [c.score for c in fused]
    assert scores == sorted(scores, reverse=True)


def test_ties_are_broken_by_chunk_id():
    """Chunk 5 first in one list and chunk 3 first in the other score the same. Without
    a tiebreaker their order would depend on which list came first - or on dict order."""
    one = reciprocal_rank_fusion([[chunk(5)], [chunk(3)]], top_k=2)
    other = reciprocal_rank_fusion([[chunk(3)], [chunk(5)]], top_k=2)

    assert ids(one) == ids(other) == [3, 5]


def test_one_ranking_alone_keeps_its_order():
    """What hybrid search returns when keyword search finds nothing: vector's order,
    untouched - even though chunk ids run the other way."""
    ranking = [chunk(9), chunk(4), chunk(6)]

    assert ids(reciprocal_rank_fusion([ranking, []], top_k=3)) == [9, 4, 6]


def test_nothing_in_gives_nothing_out():
    assert reciprocal_rank_fusion([[], []], top_k=5) == []


def test_fewer_results_than_top_k_is_not_an_error():
    assert ids(reciprocal_rank_fusion([[chunk(1)]], top_k=5)) == [1]


def test_the_chunk_carries_everything_else_unchanged():
    original = RetrievedChunk(
        chunk_id=4, document_id=2, filename="b.pdf", page_number=7, chunk_index=3,
        content="text", score=0.77,
    )

    fused = reciprocal_rank_fusion([[original]], top_k=1)[0]

    assert (fused.document_id, fused.filename, fused.page_number, fused.chunk_index,
            fused.content) == (2, "b.pdf", 7, 3, "text")
    assert fused.score == pytest.approx(1 / 61)


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(k):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[chunk(1)]], top_k=1, k=k)
