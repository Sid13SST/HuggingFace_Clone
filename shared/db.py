from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import psycopg
from psycopg_pool import ConnectionPool

from shared.config import REPO_ROOT, get_settings
from shared.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_pool() -> ConnectionPool:
    settings = get_settings()
    return ConnectionPool(settings.database_url, min_size=1, max_size=8, open=True)


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        yield conn


def migrate(schema_files: list[str | Path]) -> None:
    """Apply schema files in order.

    Every statement in these files is idempotent (CREATE ... IF NOT EXISTS), so
    this is safe to re-run. That is deliberate -- a migration tool is more
    machinery than a two-service project needs, but re-runnability is not
    optional.
    """
    with connection() as conn:
        for raw in schema_files:
            path = Path(raw)
            if not path.is_absolute():
                path = REPO_ROOT / path
            sql = path.read_text(encoding="utf-8")
            conn.execute(sql)  # type: ignore[arg-type]
            log.info("db.migrate.applied", file=str(path.relative_to(REPO_ROOT)))
        conn.commit()


def _main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "migrate":
        print("usage: python -m shared.db migrate <schema.sql> [...]", file=sys.stderr)
        return 2
    migrate(list(argv[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
