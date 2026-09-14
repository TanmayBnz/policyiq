import pytest

from policyiq.db import db_healthy, get_pool


def test_pool_connects_and_reports_healthy():
    assert db_healthy() is True


def test_chunks_table_rejects_wrong_dimension_vectors():
    """Behavioural check on the 384 constraint. Asserting on pg_attribute internals
    would couple the test to pgvector's typmod encoding; this asserts what matters."""
    with get_pool().connection() as conn:
        conn.execute("DELETE FROM documents WHERE filename = 'dimtest.pdf'")
        doc_id = conn.execute(
            "INSERT INTO documents (filename, page_count) VALUES ('dimtest.pdf', 1) RETURNING id"
        ).fetchone()[0]

        good = "[" + ",".join(["0.1"] * 384) + "]"
        conn.execute(
            "INSERT INTO chunks (document_id, page_number, chunk_index, content, embedding)"
            " VALUES (%s, 1, 0, 'ok', %s)",
            (doc_id, good),
        )

        bad = "[" + ",".join(["0.1"] * 383) + "]"
        with pytest.raises(Exception):
            conn.execute(
                "INSERT INTO chunks (document_id, page_number, chunk_index, content, embedding)"
                " VALUES (%s, 1, 1, 'bad', %s)",
                (doc_id, bad),
            )

    with get_pool().connection() as conn:
        conn.execute("DELETE FROM documents WHERE filename = 'dimtest.pdf'")
