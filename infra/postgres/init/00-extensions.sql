-- Runs once, on first cluster init.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Ledgerline and Sightline share a cluster but not a namespace.
CREATE SCHEMA IF NOT EXISTS ledgerline;
CREATE SCHEMA IF NOT EXISTS sightline;
