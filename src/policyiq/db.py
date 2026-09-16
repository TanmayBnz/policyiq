from functools import lru_cache

from psycopg_pool import ConnectionPool

from policyiq.config import settings


@lru_cache(maxsize=1)
def get_pool() -> ConnectionPool:
    """One pool per process, created lazily so that importing this module stays cheap
    and does not require a live database (which would break unit tests and CI linting)."""
    return ConnectionPool(settings.database_url, min_size=1, max_size=5, open=True)


def db_healthy() -> bool:
    try:
        with get_pool().connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def list_documents() -> list[tuple[int, str, int, int]]:
    """Every ingested document with its page and chunk counts.

    A LEFT JOIN, not an inner one: a document whose chunks failed to write has nothing
    to join against and would vanish from the listing entirely, hiding exactly the
    broken state worth seeing. It is reported with a count of zero instead.
    """
    with get_pool().connection() as conn:
        return conn.execute(
            """
            SELECT d.id, d.filename, d.page_count, count(c.id)::int AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.filename
            """
        ).fetchall()
