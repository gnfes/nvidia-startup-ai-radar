from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg import Connection

from radar_backend.core.config import get_settings


@contextmanager
def get_connection() -> Iterator[Connection]:
    """Yields a psycopg connection to the Supabase-hosted Postgres instance."""
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        yield conn
