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
