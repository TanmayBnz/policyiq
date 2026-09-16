"""Vector similarity search.

The assertions here are deliberately property-based rather than example-based. The
failure this file exists to catch is silent: pgvector's `<=>` returns cosine DISTANCE,
so ordering it the wrong way returns the least relevant chunks while the query
succeeds, rows come back, and nothing raises. An example test asserting "this question
returns that chunk" would not reliably catch it. An ordering invariant does.
"""

from pathlib import Path

import pytest

from policyiq.db import get_pool
from policyiq.ingest.pipeline import ingest_pdf
from policyiq.retrieval.vector import vector_search

OWNED_NAME = "vector-search-specimen.pdf"


@pytest.fixture(scope="module", autouse=True)
def corpus(sample_pdf: Path, tmp_path_factory):
    """Guarantee at least one known document is searchable.

    Ingested under a name these tests own, so a run cannot delete or replace a row the
    working corpus depends on. The real corpus is left in place when present — these
    assertions hold whether the table has 26 chunks or 1,570, and searching a realistic
    corpus is more meaningful than searching one document.
    """
    specimen = tmp_path_factory.mktemp("vector") / OWNED_NAME
    specimen.write_bytes(sample_pdf.read_bytes())
    ingest_pdf(specimen)
    yield
    with get_pool().connection() as conn:
        conn.execute("DELETE FROM documents WHERE filename = %s", (OWNED_NAME,))


def test_returns_the_requested_number_of_results():
    assert len(vector_search("what expenses are covered", top_k=3)) == 3


def test_results_are_ordered_best_first():
    """The invariant that catches a flipped comparison.

    Ordering by distance descending, or reporting distance where similarity is meant,
    both break this — and both are otherwise silent.
    """
    scores = [r.score for r in vector_search("what expenses are covered", top_k=5)]

    assert len(scores) == 5, "need results, or the ordering assertion below is vacuous"
    assert scores == sorted(scores, reverse=True)


def test_scores_are_similarities_not_distances():
    """Cosine similarity runs -1 to 1 with 1 meaning identical. Cosine distance runs
    0 to 2 with 0 meaning identical. Returning the latter where the former is expected
    inverts every downstream comparison."""
    results = vector_search("exclusions", top_k=5)

    assert results
    assert all(-1.0 <= r.score <= 1.0 for r in results)
    # A question drawn from the corpus should match its best chunk well. A top result
    # below this would mean the query and the documents are not in the same space.
    assert results[0].score > 0.5


def test_results_carry_everything_a_citation_needs():
    """Retrieval returning only text and a score forces citation mapping into a second
    lookup, and the two then drift. The metadata travels with the result."""
    result = vector_search("what treatments are excluded", top_k=1)[0]

    assert result.chunk_id > 0
    assert result.document_id > 0
    assert result.filename.endswith(".pdf")
    assert result.page_number >= 1
    assert result.chunk_index >= 0
    assert result.content.strip()


def test_citation_metadata_matches_the_stored_row():
    """The page number is what a user opens their PDF viewer to. If the join is wrong,
    every citation is confidently wrong and nothing looks broken."""
    result = vector_search("waiting period", top_k=1)[0]

    with get_pool().connection() as conn:
        row = conn.execute(
            "SELECT c.document_id, d.filename, c.page_number, c.chunk_index, c.content"
            " FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = %s",
            (result.chunk_id,),
        ).fetchone()

    assert row is not None, "retrieval returned a chunk_id that is not in the table"
    assert (row[0], row[1], row[2], row[3], row[4]) == (
        result.document_id,
        result.filename,
        result.page_number,
        result.chunk_index,
        result.content,
    )


def test_page_number_is_within_its_document():
    result = vector_search("claims procedure", top_k=1)[0]

    with get_pool().connection() as conn:
        page_count = conn.execute(
            "SELECT page_count FROM documents WHERE id = %s", (result.document_id,)
        ).fetchone()[0]

    assert 1 <= result.page_number <= page_count


def test_no_chunk_is_returned_twice():
    results = vector_search("hospitalisation", top_k=10)
    ids = [r.chunk_id for r in results]

    assert len(ids) == len(set(ids))


def test_the_question_actually_steers_the_result():
    """Proves the embedding is being used at all.

    Two unrelated questions must not produce the same ranking. If the query vector were
    ignored — or the same vector sent for every question — this is what would expose it.
    """
    covered = vector_search("what is the waiting period for pre-existing diseases", top_k=5)
    claims = vector_search("how do I notify the company of a claim", top_k=5)

    assert [r.chunk_id for r in covered] != [r.chunk_id for r in claims]


def test_a_larger_top_k_extends_the_same_ranking():
    """Asking for more results must not reshuffle the ones already agreed on."""
    narrow = vector_search("exclusions apply to", top_k=3)
    wide = vector_search("exclusions apply to", top_k=6)

    assert [r.chunk_id for r in wide][:3] == [r.chunk_id for r in narrow]


def test_asking_for_more_than_the_corpus_holds_is_not_an_error():
    with get_pool().connection() as conn:
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    results = vector_search("policy", top_k=total + 50)

    assert len(results) == total
