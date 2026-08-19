-- Ledgerline storage. Idempotent: safe to re-run.
--
-- Design note: every retrievable unit keeps a *resolvable span* -- a page plus
-- bounding box, a table cell address, or an audio time range. Without that the
-- citation verifier has nothing to check against, and the whole provenance
-- story collapses into "the LLM said so".

CREATE SCHEMA IF NOT EXISTS ledgerline;

CREATE TABLE IF NOT EXISTS ledgerline.issuers (
    cik           text PRIMARY KEY,
    ticker        text,
    name          text NOT NULL,
    sic           text,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ledgerline.documents (
    id            bigserial PRIMARY KEY,
    cik           text NOT NULL REFERENCES ledgerline.issuers(cik) ON DELETE CASCADE,
    -- 'filing' | 'transcript' | 'deck'
    kind          text NOT NULL,
    form          text,                     -- 10-K, 10-Q, 8-K, ...
    accession     text,                     -- EDGAR accession number, when applicable
    title         text,
    source_url    text,
    filed_at      date,
    period_end    date,
    fiscal_period text,                     -- e.g. FY2025, Q3-2025
    -- Scale stated by the document itself ("in thousands" -> 1000). Numeric
    -- answers are normalised through this before comparison.
    scale_hint    numeric NOT NULL DEFAULT 1,
    raw_path      text,
    ingested_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (cik, kind, accession)
);

CREATE INDEX IF NOT EXISTS documents_cik_period_idx
    ON ledgerline.documents (cik, period_end DESC);

-- Retrievable text. One row per chunk, with the span that produced it.
CREATE TABLE IF NOT EXISTS ledgerline.chunks (
    id            bigserial PRIMARY KEY,
    document_id   bigint NOT NULL REFERENCES ledgerline.documents(id) ON DELETE CASCADE,
    ordinal       int NOT NULL,
    content       text NOT NULL,
    -- provenance, exactly one of these is populated per chunk kind
    page          int,
    bbox          numeric[],                -- [x0, y0, x1, y1] in PDF points
    char_start    int,
    char_end      int,
    audio_start_s numeric,
    audio_end_s   numeric,
    speaker       text,                     -- from diarization: 'CFO', 'analyst', ...
    section       text,                     -- 'Item 1A', 'Q&A', 'MD&A'
    -- 256 dims: model2vec potion-base-8M, the embedder the baselines were
    -- measured with. Changing model means changing this and reindexing;
    -- tests/test_embeddings.py pins the two together so they cannot drift.
    embedding     vector(256),
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON ledgerline.chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON ledgerline.chunks (document_id);
-- HNSW over cosine distance. m/ef_construction are the two knobs worth an
-- ablation in the report; these are the defaults we measured from.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON ledgerline.chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Tables are stored as data, never as flattened prose. This is the single
-- decision that moves numeric accuracy the most: the agent queries `cells`
-- instead of reading digits out of a rendered string.
CREATE TABLE IF NOT EXISTS ledgerline.tables (
    id            bigserial PRIMARY KEY,
    document_id   bigint NOT NULL REFERENCES ledgerline.documents(id) ON DELETE CASCADE,
    ordinal       int NOT NULL,
    caption       text,
    page          int,
    bbox          numeric[],
    n_rows        int NOT NULL,
    n_cols        int NOT NULL,
    scale_hint    numeric NOT NULL DEFAULT 1,
    unit          text,                     -- 'USD', 'shares', 'percent'
    UNIQUE (document_id, ordinal)
);

CREATE TABLE IF NOT EXISTS ledgerline.table_cells (
    table_id      bigint NOT NULL REFERENCES ledgerline.tables(id) ON DELETE CASCADE,
    row_idx       int NOT NULL,
    col_idx       int NOT NULL,
    raw           text,
    value_num     numeric,                  -- parsed and scale-normalised
    is_header     boolean NOT NULL DEFAULT false,
    PRIMARY KEY (table_id, row_idx, col_idx)
);

CREATE INDEX IF NOT EXISTS table_cells_numeric_idx
    ON ledgerline.table_cells (table_id) WHERE value_num IS NOT NULL;

-- Agent runs, for replay. A run you cannot replay is a bug you cannot fix.
CREATE TABLE IF NOT EXISTS ledgerline.runs (
    id            uuid PRIMARY KEY,
    question      text NOT NULL,
    cik           text,
    answer        text,
    citations     jsonb NOT NULL DEFAULT '[]'::jsonb,
    state         jsonb NOT NULL DEFAULT '{}'::jsonb,
    -- 'answered' | 'refused' | 'degraded'
    outcome       text,
    prompt_version text,
    model         text,
    input_tokens  int,
    output_tokens int,
    cost_usd      numeric,
    latency_ms    int,
    trace_id      text,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS runs_created_idx ON ledgerline.runs (created_at DESC);

-- Reciprocal rank fusion of dense and lexical retrieval.
--
-- RRF beats score normalisation here because BM25 and cosine live on
-- incomparable scales; fusing on *rank* sidesteps the calibration problem
-- entirely. rrf_k=60 is the value from the original paper and the one the
-- baseline was measured with -- change it and re-run the suite.
CREATE OR REPLACE FUNCTION ledgerline.hybrid_search(
    p_query      text,
    p_embedding  vector(256),
    p_cik        text DEFAULT NULL,
    p_limit      int DEFAULT 50,
    p_rrf_k      int DEFAULT 60
)
RETURNS TABLE (chunk_id bigint, document_id bigint, content text, score double precision)
LANGUAGE sql STABLE AS $$
    WITH dense AS (
        SELECT c.id,
               row_number() OVER (ORDER BY c.embedding <=> p_embedding) AS rank
        FROM ledgerline.chunks c
        JOIN ledgerline.documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
          AND (p_cik IS NULL OR d.cik = p_cik)
        ORDER BY c.embedding <=> p_embedding
        LIMIT p_limit * 4
    ),
    lexical AS (
        SELECT c.id,
               row_number() OVER (
                   ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('english', p_query)) DESC
               ) AS rank
        FROM ledgerline.chunks c
        JOIN ledgerline.documents d ON d.id = c.document_id
        WHERE c.tsv @@ websearch_to_tsquery('english', p_query)
          AND (p_cik IS NULL OR d.cik = p_cik)
        LIMIT p_limit * 4
    ),
    fused AS (
        SELECT COALESCE(dense.id, lexical.id) AS id,
               COALESCE(1.0 / (p_rrf_k + dense.rank), 0.0)
             + COALESCE(1.0 / (p_rrf_k + lexical.rank), 0.0) AS score
        FROM dense
        FULL OUTER JOIN lexical ON dense.id = lexical.id
    )
    SELECT c.id, c.document_id, c.content, fused.score
    FROM fused
    JOIN ledgerline.chunks c ON c.id = fused.id
    ORDER BY fused.score DESC
    LIMIT p_limit;
$$;
