from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector

from .config import get_settings


def psycopg_database_url() -> str:
    """Return a URL accepted by psycopg.

    SQLAlchemy uses the ``postgresql+psycopg://`` prefix, while psycopg's
    direct connection API expects ``postgresql://``.
    """

    return get_settings().database_url.replace("postgresql+psycopg://", "postgresql://", 1)


@contextmanager
def database_connection() -> Iterator[psycopg.Connection]:
    with psycopg.connect(psycopg_database_url()) as connection:
        register_vector(connection)
        yield connection

