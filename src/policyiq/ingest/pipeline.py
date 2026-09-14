from pathlib import Path

from pgvector.psycopg import register_vector

from policyiq.config import settings
from policyiq.db import get_pool
from policyiq.embeddings import embed_texts
from policyiq.ingest.chunking import chunk_pages
from policyiq.ingest.pdf import extract_pages
from policyiq.ingest.validation import check_document
from policyiq.schemas import IngestResult

INSERT_CHUNK = """
    INSERT INTO chunks (document_id, page_number, chunk_index, content, embedding)
    VALUES (%s, %s, %s, %s, %s)
"""


def ingest_pdf(pdf_path: Path) -> IngestResult:
    """Parse, chunk, embed and store one PDF.

    Re-ingesting a filename replaces the previous version rather than adding to it.
    Development means running ingestion repeatedly, and an append-only pipeline would
    fill the corpus with duplicates that crowd out genuine matches in retrieval.

    All the slow work happens before the database connection is opened. That ordering
    is deliberate on two counts: a pooled connection held open for the length of an
    embedding run starves the other five, and deleting the existing document before
    knowing the replacement can be produced would turn a parse failure into data loss.
    """
    filename = pdf_path.name

    # pypdf raises a family of unrelated exception types for a file it cannot read:
    # an encrypted document, a truncated one, and an HTML error page saved with a .pdf
    # name all fail differently. Every one of them is a bad upload rather than a fault
    # in this service, so they are caught together and reported as a rejection.
    # Encrypted documents are rejected rather than decrypted on purpose - the
    # decryption dependency is weight the container does not otherwise need.
    try:
        pages = extract_pages(pdf_path)
    except Exception as exc:
        raise ValueError(
            f"{filename} rejected at intake - unreadable: {type(exc).__name__}: {exc}"
        ) from exc

    # Intake runs before any work is done and before anything is written. A document
    # that gets past this point is competing for space in every future set of search
    # results, and its failure mode is silent.
    rejections = check_document(pages)
    if rejections:
        reasons = "; ".join(f"{r.rule}: {r.detail}" for r in rejections)
        raise ValueError(f"{filename} rejected at intake - {reasons}")

    chunks = chunk_pages(pages, settings.chunk_target_chars, settings.chunk_overlap_chars)
    vectors = embed_texts([chunk.content for chunk in chunks])

    with get_pool().connection() as conn:
        # Teaches this connection to send a Python list as a pgvector `vector` rather
        # than a Postgres array, which the column would reject.
        register_vector(conn)

        # Replacement is a delete followed by an insert inside one transaction: psycopg
        # commits when this block exits cleanly and rolls back if it raises, so a
        # failure part-way through cannot leave the corpus missing a document. Chunks
        # are removed by ON DELETE CASCADE, so no orphans are possible.
        conn.execute("DELETE FROM documents WHERE filename = %s", (filename,))
        document_id = conn.execute(
            "INSERT INTO documents (filename, page_count) VALUES (%s, %s) RETURNING id",
            (filename, len(pages)),
        ).fetchone()[0]

        with conn.cursor() as cur:
            cur.executemany(
                INSERT_CHUNK,
                [
                    (document_id, chunk.page_number, chunk.chunk_index, chunk.content, vector)
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )

    return IngestResult(
        document_id=document_id,
        filename=filename,
        page_count=len(pages),
        chunk_count=len(chunks),
    )
