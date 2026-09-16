from dataclasses import dataclass

from pgvector import Vector
from pgvector.psycopg import register_vector

from policyiq.db import get_pool
from policyiq.embeddings import embed_query


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    filename: str
    page_number: int
    chunk_index: int
    content: str
    score: float


def vector_search(question: str, top_k: int) -> list[RetrievedChunk]:
    """Rank chunks by cosine similarity to the question.

    pgvector's `<=>` operator returns cosine DISTANCE, where 0 means identical.
    Similarity is 1 - distance, so ordering by distance ascending is the same as
    ordering by similarity descending. Getting this backwards returns the least
    relevant chunks and is silent — nothing errors, the answers are just wrong.
    """
    # BUGFIX: this was `embedding = embed_query(question)` passed straight through as a
    # plain list, which failed with "operator does not exist: vector <=> double
    # precision[]". psycopg adapts a Python list to a Postgres float8[], and there is no
    # operator taking vector on the left and float8[] on the right.
    #
    # The confusing part is that the ingest pipeline passes a plain list and works. The
    # difference is that an INSERT has a declared target column of type vector(384), so
    # Postgres applies an assignment cast. Here the parameter has no declared target, so
    # its type is inferred from what was sent and operator resolution then finds nothing.
    # "It worked on insert" says nothing about whether it works inside an expression.
    #
    # Wrapping in Vector is what register_vector below is for: it registers a dumper for
    # this type. Without the wrapper that call was doing nothing. `%s::vector` in the SQL
    # would also work and would make register_vector redundant; the wrapper is preferred
    # because it states the intent where the parameter is built.
    embedding = Vector(embed_query(question))
    with get_pool().connection() as conn:
        register_vector(conn)
        rows = conn.execute(
            """
            SELECT c.id, c.document_id, d.filename, c.page_number, c.chunk_index,
                   c.content, 1 - (c.embedding <=> %s) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            -- The c.id tiebreaker makes the ordering total rather than partial.
            -- Distance alone leaves tied chunks in an order Postgres may choose
            -- differently for LIMIT 3 than for LIMIT 6, because the two pick different
            -- plans - so the same question returned a different ranking run to run.
            -- Exact ties are real here: nine corpus documents are the same standardised
            -- product, so identical clause text yields identical embeddings.
            -- This matters most for evaluation. Context recall asks whether the right
            -- chunk is in the top k; if ranking wobbles on tied scores, the score moves
            -- for no reason and a real regression is indistinguishable from noise.
            ORDER BY c.embedding <=> %s, c.id
            LIMIT %s
            """,
            (embedding, embedding, top_k),
        ).fetchall()

    return [
        RetrievedChunk(
            chunk_id=r[0], document_id=r[1], filename=r[2], page_number=r[3],
            chunk_index=r[4], content=r[5], score=float(r[6]),
        )
        for r in rows
    ]
