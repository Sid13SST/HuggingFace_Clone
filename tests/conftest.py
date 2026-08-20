"""Shared fixtures.

The database fixture lives here because two test modules need it and the rule
it encodes is a project-wide one: **never fail a test run for want of
infrastructure.** A developer without Docker running should get skips and a
reason, not a wall of connection errors that buries the failures they actually
caused.

The other half of that rule is that skips must not become a hiding place, which
is why CI's schema job runs these same modules against a real Postgres. A skip
is acceptable locally and unacceptable in the job that has a database.
"""

from __future__ import annotations

import os

import pytest


def database_or_skip():
    """Connect to the configured database, or skip with the reason why."""
    if os.environ.get("LEDGERLINE_SKIP_DB_TESTS"):
        pytest.skip("LEDGERLINE_SKIP_DB_TESTS is set")
    try:
        import psycopg

        from shared.config import get_settings

        return psycopg.connect(get_settings().database_url, connect_timeout=3)
    except Exception as exc:  # noqa: BLE001 - any failure to reach a db is a skip
        pytest.skip(f"no database: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="module")
def db():
    conn = database_or_skip()
    try:
        yield conn
    finally:
        conn.close()
