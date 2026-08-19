-- Sightline storage. Idempotent: safe to re-run.
--
-- The load-bearing idea: reports, imagery and defects all live in one
-- Postgres, so "same road segment AND visually similar" is a single indexed
-- query rather than a spatial store talking to a vector store over the
-- network and hoping they agree.

CREATE SCHEMA IF NOT EXISTS sightline;

-- Road segments from OpenStreetMap. Snapping to a segment beats a naive
-- radius: two potholes 25 m apart on opposite sides of a divided road are
-- different work orders, and a radius cannot tell you that.
CREATE TABLE IF NOT EXISTS sightline.segments (
    id            bigserial PRIMARY KEY,
    osm_way_id    bigint UNIQUE,
    name          text,
    highway       text,
    geom          geography(LineString, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS segments_geom_idx ON sightline.segments USING gist (geom);

-- Raw citizen reports, as received. Never mutated -- dedupe writes to
-- `defects` and links back, so a wrong merge is reversible.
CREATE TABLE IF NOT EXISTS sightline.reports (
    id            bigserial PRIMARY KEY,
    source        text NOT NULL,            -- 'open311', 'socrata:chicago', ...
    external_id   text NOT NULL,
    category      text,
    description   text,
    reported_at   timestamptz,
    status        text,
    geom          geography(Point, 4326),
    segment_id    bigint REFERENCES sightline.segments(id) ON DELETE SET NULL,
    text_embedding vector(384),
    raw           jsonb NOT NULL DEFAULT '{}'::jsonb,
    ingested_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source, external_id)
);

CREATE INDEX IF NOT EXISTS reports_geom_idx ON sightline.reports USING gist (geom);
CREATE INDEX IF NOT EXISTS reports_segment_idx ON sightline.reports (segment_id);
CREATE INDEX IF NOT EXISTS reports_reported_at_idx ON sightline.reports (reported_at DESC);
CREATE INDEX IF NOT EXISTS reports_text_embedding_idx
    ON sightline.reports USING hnsw (text_embedding vector_cosine_ops);

-- Street-level frames pulled for a reported location.
CREATE TABLE IF NOT EXISTS sightline.images (
    id             bigserial PRIMARY KEY,
    source         text NOT NULL DEFAULT 'mapillary',
    external_id    text NOT NULL,
    captured_at    timestamptz,
    geom           geography(Point, 4326),
    compass_angle  numeric,
    width          int,
    height         int,
    storage_path   text,
    image_embedding vector(512),
    UNIQUE (source, external_id)
);

CREATE INDEX IF NOT EXISTS images_geom_idx ON sightline.images USING gist (geom);
CREATE INDEX IF NOT EXISTS images_captured_idx ON sightline.images (captured_at DESC);
CREATE INDEX IF NOT EXISTS images_embedding_idx
    ON sightline.images USING hnsw (image_embedding vector_cosine_ops);

-- One row per model output on one frame. `score` is kept raw and uncalibrated
-- here; calibration is applied at read time so a recalibration does not
-- require reprocessing imagery.
CREATE TABLE IF NOT EXISTS sightline.detections (
    id            bigserial PRIMARY KEY,
    image_id      bigint NOT NULL REFERENCES sightline.images(id) ON DELETE CASCADE,
    model         text NOT NULL,
    model_version text NOT NULL,
    label         text NOT NULL,
    score         real NOT NULL,
    bbox          real[] NOT NULL,          -- [x1, y1, x2, y2] in pixels
    mask_area_px  int,
    -- Physical estimates, from monocular depth x mask area. Severity keys off
    -- these, not off an adjective from a language model.
    depth_m       numeric,
    extent_cm     numeric,
    extent_m2     numeric,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS detections_image_idx ON sightline.detections (image_id);
CREATE INDEX IF NOT EXISTS detections_label_score_idx ON sightline.detections (label, score DESC);

-- The deduplicated work queue. This is the product.
CREATE TABLE IF NOT EXISTS sightline.defects (
    id            bigserial PRIMARY KEY,
    segment_id    bigint REFERENCES sightline.segments(id) ON DELETE SET NULL,
    label         text NOT NULL,
    geom          geography(Point, 4326) NOT NULL,
    severity      text NOT NULL,            -- 'routine' | 'priority' | 'urgent'
    extent_cm     numeric,
    confidence    real,
    -- Cited clause from the municipal standards corpus. An uncited severity
    -- band is an opinion; a cited one is a decision a supervisor can audit.
    standard_clause text,
    sla_days      int,
    status        text NOT NULL DEFAULT 'open',
    first_seen_at timestamptz,
    last_seen_at  timestamptz,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS defects_geom_idx ON sightline.defects USING gist (geom);
CREATE INDEX IF NOT EXISTS defects_open_idx ON sightline.defects (severity, created_at DESC)
    WHERE status = 'open';

CREATE TABLE IF NOT EXISTS sightline.defect_reports (
    defect_id     bigint NOT NULL REFERENCES sightline.defects(id) ON DELETE CASCADE,
    report_id     bigint NOT NULL REFERENCES sightline.reports(id) ON DELETE CASCADE,
    PRIMARY KEY (defect_id, report_id)
);

-- Human overrides. This table is the flywheel: every row is a labelled
-- training example the model got wrong, with the reason attached.
CREATE TABLE IF NOT EXISTS sightline.reviews (
    id             bigserial PRIMARY KEY,
    defect_id      bigint NOT NULL REFERENCES sightline.defects(id) ON DELETE CASCADE,
    reviewer       text NOT NULL,
    action         text NOT NULL,           -- 'accept' | 'override' | 'reject'
    field          text,                    -- which decision was overridden
    proposed       text,
    corrected      text,
    reason         text,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS reviews_defect_idx ON sightline.reviews (defect_id);
CREATE INDEX IF NOT EXISTS reviews_action_idx ON sightline.reviews (action, created_at DESC);

-- Candidate duplicate pairs: spatial blocking first, then semantic scoring.
--
-- Order matters for cost. The GiST index cuts the candidate set to metres-
-- apart pairs before any vector maths happens, which is what keeps this
-- tractable at hundreds of thousands of reports. Doing it the other way --
-- nearest-neighbour over every embedding, then filtering by distance -- is
-- the version that looks fine on 1k rows and falls over on 100k.
CREATE OR REPLACE FUNCTION sightline.duplicate_candidates(
    p_report_id      bigint,
    p_radius_m       double precision DEFAULT 30,
    p_window_days    int DEFAULT 45,
    p_limit          int DEFAULT 25
)
RETURNS TABLE (
    report_id        bigint,
    distance_m       double precision,
    same_segment     boolean,
    text_similarity  double precision,
    combined         double precision
)
LANGUAGE sql STABLE AS $$
    WITH target AS (
        SELECT id, geom, segment_id, text_embedding, reported_at, category
        FROM sightline.reports WHERE id = p_report_id
    )
    SELECT r.id,
           ST_Distance(r.geom, t.geom) AS distance_m,
           (r.segment_id IS NOT DISTINCT FROM t.segment_id) AS same_segment,
           CASE
               WHEN r.text_embedding IS NULL OR t.text_embedding IS NULL THEN NULL
               ELSE 1 - (r.text_embedding <=> t.text_embedding)
           END AS text_similarity,
           -- Weights are the tuned values from evals/baselines; changing them
           -- without re-running sightline.dedupe is how a silent regression
           -- gets shipped.
           0.45 * (1 - LEAST(ST_Distance(r.geom, t.geom) / p_radius_m, 1))
         + 0.35 * COALESCE(1 - (r.text_embedding <=> t.text_embedding), 0)
         + 0.20 * (CASE WHEN r.segment_id IS NOT DISTINCT FROM t.segment_id THEN 1 ELSE 0 END)
           AS combined
    FROM sightline.reports r, target t
    WHERE r.id <> t.id
      AND r.geom IS NOT NULL
      AND ST_DWithin(r.geom, t.geom, p_radius_m)
      AND (r.reported_at IS NULL OR t.reported_at IS NULL
           OR abs(extract(epoch FROM r.reported_at - t.reported_at)) <= p_window_days * 86400)
      AND (r.category IS NOT DISTINCT FROM t.category OR t.category IS NULL)
    ORDER BY combined DESC
    LIMIT p_limit;
$$;

-- Open queue for the reviewer UI, worst first.
CREATE OR REPLACE VIEW sightline.open_queue AS
SELECT d.id,
       d.label,
       d.severity,
       d.extent_cm,
       d.confidence,
       d.sla_days,
       d.standard_clause,
       s.name AS street,
       count(dr.report_id) AS report_count,
       d.first_seen_at,
       d.last_seen_at
FROM sightline.defects d
LEFT JOIN sightline.segments s ON s.id = d.segment_id
LEFT JOIN sightline.defect_reports dr ON dr.defect_id = d.id
WHERE d.status = 'open'
GROUP BY d.id, s.name
ORDER BY CASE d.severity WHEN 'urgent' THEN 0 WHEN 'priority' THEN 1 ELSE 2 END,
         count(dr.report_id) DESC,
         d.first_seen_at;
