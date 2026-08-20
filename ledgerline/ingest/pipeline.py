"""Chunk, embed, and write documents into Postgres.

The point of this module is that the retrieval numbers in the README should
describe the system the schema describes. Until something wrote embeddings into
`ledgerline.chunks`, `ledgerline.hybrid_search` had never ranked a real row and
every measurement described the offline Python mirror instead.

Two decisions worth stating:

Re-ingesting a document *replaces* its chunks rather than upserting them. An
upsert keyed on (document_id, ordinal) leaves orphans behind whenever a
document shrinks -- re-chunk a filing into 40 pieces where it used to be 60 and
the last 20 stale chunks stay in the index, retrievable and wrong. Replace has
no such failure mode, and at filing scale the churn is irrelevant.

`external_id` carries the identity a chunk has *outside* the database. Without
it a SQL ranking and an offline ranking are two lists of unrelated integers and
cannot be compared at all, which is the whole reason the column exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import psycopg

from shared.embeddings import Embedder
from shared.logging import get_logger

log = get_logger(__name__)


def as_vector_literal(vector: np.ndarray) -> str:
    """Render a vector for `%s::vector`.

    pgvector ships a psycopg adapter that would send these in binary, which is
    faster and would let numpy arrays be passed straight through. It is not
    used, for one reason: the *read* path already casts a plain list this way,
    and an adapter has to be registered per connection. A write path that
    silently depends on someone having called `register_vector` on this
    particular connection is a bug waiting for the first pooled caller.

    One format, both directions, no per-connection setup. If ingest volume ever
    makes the text encoding matter, that is the moment to add the adapter -- and
    to measure it rather than assume.
    """
    return "[" + ",".join(f"{float(x):.9g}" for x in vector) + "]"


@dataclass(frozen=True)
class Issuer:
    cik: str
    name: str
    ticker: str | None = None
    sic: str | None = None


@dataclass(frozen=True)
class Document:
    cik: str
    kind: str  # 'filing' | 'transcript' | 'deck'
    accession: str
    form: str | None = None
    title: str | None = None
    source_url: str | None = None
    fiscal_period: str | None = None
    #: Scale the document states about itself ("in thousands" -> 1000).
    scale_hint: float = 1.0


@dataclass(frozen=True)
class ChunkRow:
    """One retrievable unit, with the provenance the citation verifier needs."""

    external_id: str
    content: str
    ordinal: int
    section: str | None = None
    speaker: str | None = None
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None


@dataclass
class IngestResult:
    documents: int = 0
    chunks: int = 0
    replaced: int = 0
    external_ids: list[str] = field(default_factory=list)


def upsert_issuer(conn: psycopg.Connection, issuer: Issuer) -> None:
    conn.execute(
        """
        INSERT INTO ledgerline.issuers (cik, ticker, name, sic)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (cik) DO UPDATE
            SET ticker = EXCLUDED.ticker,
                name   = EXCLUDED.name,
                sic    = EXCLUDED.sic
        """,
        (issuer.cik, issuer.ticker, issuer.name, issuer.sic),
    )


def upsert_document(conn: psycopg.Connection, document: Document) -> int:
    """Insert or update a document, returning its id.

    `ON CONFLICT ... DO UPDATE` rather than `DO NOTHING` because DO NOTHING
    returns no row, and then the caller has to issue a second SELECT to find
    the id it just failed to insert -- a race in anything concurrent.
    """
    row = conn.execute(
        """
        INSERT INTO ledgerline.documents
            (cik, kind, form, accession, title, source_url, fiscal_period, scale_hint)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (cik, kind, accession) DO UPDATE
            SET form          = EXCLUDED.form,
                title         = EXCLUDED.title,
                source_url    = EXCLUDED.source_url,
                fiscal_period = EXCLUDED.fiscal_period,
                scale_hint    = EXCLUDED.scale_hint
        RETURNING id
        """,
        (
            document.cik,
            document.kind,
            document.form,
            document.accession,
            document.title,
            document.source_url,
            document.fiscal_period,
            document.scale_hint,
        ),
    ).fetchone()
    assert row is not None  # RETURNING on a DO UPDATE always yields a row
    return int(row[0])


def write_chunks(
    conn: psycopg.Connection,
    document_id: int,
    chunks: Sequence[ChunkRow],
    embeddings: np.ndarray,
) -> int:
    """Replace this document's chunks with `chunks`. Returns rows removed."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"{len(chunks)} chunks but {len(embeddings)} embeddings -- "
            "a chunk written without its vector is invisible to dense retrieval"
        )

    removed = conn.execute(
        "DELETE FROM ledgerline.chunks WHERE document_id = %s", (document_id,)
    ).rowcount

    with conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO ledgerline.chunks
                (document_id, ordinal, content, external_id, section, speaker,
                 page, char_start, char_end, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            """,
            [
                (
                    document_id,
                    chunk.ordinal,
                    chunk.content,
                    chunk.external_id,
                    chunk.section,
                    chunk.speaker,
                    chunk.page,
                    chunk.char_start,
                    chunk.char_end,
                    as_vector_literal(vector),
                )
                for chunk, vector in zip(chunks, embeddings, strict=True)
            ],
        )
    return int(removed)


def ingest_document(
    conn: psycopg.Connection,
    issuer: Issuer,
    document: Document,
    chunks: Sequence[ChunkRow],
    embedder: Embedder,
) -> IngestResult:
    """Write one document and its chunks. Caller owns the transaction.

    Leaving the commit to the caller is what makes a multi-document ingest
    atomic: a filing that fails halfway through should not leave half its
    chunks searchable.
    """
    upsert_issuer(conn, issuer)
    document_id = upsert_document(conn, document)

    embeddings = (
        embedder.encode([c.content for c in chunks])
        if chunks
        else np.zeros((0, getattr(embedder, "dim", 1)), dtype=np.float32)
    )
    replaced = write_chunks(conn, document_id, chunks, embeddings)

    log.info(
        "ingest.document",
        cik=issuer.cik,
        accession=document.accession,
        chunks=len(chunks),
        replaced=replaced,
    )
    return IngestResult(
        documents=1,
        chunks=len(chunks),
        replaced=replaced,
        external_ids=[c.external_id for c in chunks],
    )
