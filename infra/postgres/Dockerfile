# Both projects need one Postgres: Ledgerline wants pgvector + full-text search,
# Sightline wants pgvector + PostGIS in the same query. No public image ships
# both, so build it: PostGIS base, pgvector added from PGDG.
FROM postgis/postgis:16-3.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*
