-- Keyword search. Runs after 001 on a fresh volume (the entrypoint applies files in
-- name order). An existing database needs it applied by hand, once:
--
--   docker compose exec -T postgres psql -U policyiq -d policyiq < sql/002_keyword_search.sql
--
-- Both statements are safe to repeat, and neither needs a re-ingest: adding a STORED
-- generated column computes it for every existing row.

-- A tsvector is the chunk reduced to its searchable words: lower-cased, stemmed
-- ("hospitalisation" and "hospitalised" both become "hospitalis"), and with common
-- English words such as "the" and "is" dropped. GENERATED means Postgres keeps it in step
-- with `content`, so ingestion does not have to remember to fill it in.
--
-- The 'english' configuration is spelled out. to_tsvector without it uses a server
-- setting, which can differ between machines - and a generated column must be
-- deterministic, so Postgres rejects the one-argument form here anyway.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- The filename's words, searchable alongside every chunk of that document. This is the
-- largest single gain measured: the golden set's questions name products ("HDFC ERGO
-- Optima Secure", "Arogya Sanjeevani ... 12th December 2022") and often only the
-- filename says which document a page belongs to. Page text rarely repeats the product
-- name, and one case's distinguishing date appears nowhere but the filename.
--
-- Punctuation becomes spaces first. Left alone, the parser reads
-- "hdfc-ergo-optima-secure.pdf" as a single file-path token and none of the product
-- words become searchable.
--
-- It lives on documents, not chunks, because a generated column can only read its own
-- row. The search query joins the two.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS filename_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', translate(filename, '-_.()', '     '))) STORED;

-- Deliberately no GIN index, for the same reason 001 has no vector index. The search
-- matches against content_tsv || filename_tsv, an expression spanning two tables, which
-- no index on either table can serve - so an index would be built, maintained, and never
-- used. At 1,570 chunks the scan is measured in milliseconds (see docs/measurements.md).
-- The route at scale is to copy the filename's words into a chunks column at ingest time
-- and index that single column.
