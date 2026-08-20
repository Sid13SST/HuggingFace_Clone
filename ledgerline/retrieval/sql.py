"""Retrieval against Postgres. The production path.

`HybridRetriever` is the offline mirror: it exists so fusion can be tuned and
gated without a database. This is the thing that actually runs, and the two are
only worth having if their differences are known rather than assumed.

They are not identical, and should not be expected to be:

  * The **dense** arms must agree exactly. Both read the same committed vectors
    and both order by cosine, so any disagreement is a bug -- a vector written
    wrong, a dimension mismatch, a normalisation applied on one side only.
  * The **lexical** arms genuinely differ. Offline is BM25 with our own
    tokeniser and stopword list; Postgres is `ts_rank_cd` over a snowball-
    stemmed tsvector. These are different ranking functions over different
    tokenisations, and forcing them to agree would mean crippling one.

So the dense arm gets an equality assertion and the lexical arm gets a measured
divergence. Reporting a number for the second is honest; asserting equality
would just be a test tuned until it passed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import psycopg

from ledgerline.retrieval.hybrid import RRF_K
from shared.embeddings import Embedder, normalize

#: Per-arm candidate depth. Mirrors HybridRetriever.candidate_k and is passed
#: to hybrid_search as p_candidate_k, so the two paths search to one depth.
CANDIDATE_K = 40


@dataclass
class SqlRetriever:
    """Thin, honest wrapper over `ledgerline.hybrid_search` and its two arms.

    Holds a connection rather than a pool because every caller so far is either
    a CLI command or a test, both of which want one session and explicit
    transaction boundaries.
    """

    conn: psycopg.Connection
    embedder: Embedder
    cik: str | None = None
    rrf_k: int = RRF_K
    candidate_k: int = CANDIDATE_K

    def _encode(self, query: str) -> list[float]:
        """Encode and row-normalise, matching the offline path exactly.

        Cosine distance is scale invariant so this cannot change an ordering --
        it is here so that a stored vector and a query vector are produced by
        one code path, and a future switch to `<#>` (inner product, which is
        *not* scale invariant) does not silently change results.
        """
        return [float(x) for x in normalize(self.embedder.encode([query]))[0]]

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        rows = self.conn.execute(
            """
            SELECT c.external_id, h.score
            FROM ledgerline.hybrid_search(%s, %s::vector, %s, %s, %s, %s) h
            JOIN ledgerline.chunks c ON c.id = h.chunk_id
            ORDER BY h.score DESC, c.external_id
            """,
            (query, self._encode(query), self.cik, k, self.rrf_k, self.candidate_k),
        ).fetchall()
        return [(str(external_id), float(score)) for external_id, score in rows]

    def rank(self, query: str, k: int = 10) -> list[str]:
        return [external_id for external_id, _ in self.search(query, k)]

    def dense_rank(self, query: str, k: int = 10) -> list[str]:
        """The vector arm alone, unfused. Compared for equality offline."""
        rows = self.conn.execute(
            """
            SELECT c.external_id
            FROM ledgerline.chunks c
            JOIN ledgerline.documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
              AND (%s::text IS NULL OR d.cik = %s)
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (self.cik, self.cik, self._encode(query), k),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def lexical_rank(self, query: str, k: int = 10) -> list[str]:
        """The tsvector arm alone. Compared for *divergence*, not equality."""
        rows = self.conn.execute(
            """
            SELECT c.external_id
            FROM ledgerline.chunks c
            JOIN ledgerline.documents d ON d.id = c.document_id
            WHERE c.tsv @@ ledgerline.any_lexeme_tsquery(%s)
              AND (%s::text IS NULL OR d.cik = %s)
            ORDER BY ts_rank_cd(c.tsv, ledgerline.any_lexeme_tsquery(%s)) DESC,
                     c.external_id
            LIMIT %s
            """,
            (query, self.cik, self.cik, query, k),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def indexed_ids(self) -> set[str]:
        """External ids currently searchable. A corpus/database drift check."""
        rows = self.conn.execute(
            """
            SELECT c.external_id
            FROM ledgerline.chunks c
            WHERE c.external_id IS NOT NULL AND c.embedding IS NOT NULL
            """
        ).fetchall()
        return {str(row[0]) for row in rows}


def explain_plan(conn: psycopg.Connection, query: str, embedding: Sequence[float]) -> str:
    """`EXPLAIN ANALYZE` for the dense arm.

    Here because "we added an HNSW index" is worth exactly nothing without
    evidence the planner uses it -- an index that loses to a sequential scan at
    fixture scale is fine, but it should be a known fact rather than a hope.
    """
    rows = conn.execute(
        """
        EXPLAIN ANALYZE
        SELECT c.id FROM ledgerline.chunks c
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> %s::vector
        LIMIT 10
        """,
        ([float(x) for x in embedding],),
    ).fetchall()
    return "\n".join(str(row[0]) for row in rows)
