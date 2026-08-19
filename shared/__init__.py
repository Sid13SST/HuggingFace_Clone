"""Infrastructure shared by Ledgerline and Sightline.

Deliberately thin: config, an HTTP client that respects upstream rate limits,
a Postgres helper, and the eval harness. Anything project-specific belongs in
the project package, not here.
"""

__all__ = ["config", "db", "evals", "http", "logging"]
