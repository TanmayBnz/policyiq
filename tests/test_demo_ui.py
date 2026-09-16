"""The demo page and the endpoint that backs it.

The page is served by the API itself rather than hosted anywhere. That is not a
limitation to apologise for: embedding and generation both run on this machine, and
the claim that no document text leaves it is the point of the project. A hosted demo
would send insurer PDFs to someone else's database and someone else's model.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from policyiq.db import get_pool
from policyiq.ingest.pipeline import ingest_pdf
from policyiq.main import app

client = TestClient(app)

OWNED_NAME = "demo-specimen.pdf"


@pytest.fixture
def ingested(sample_pdf: Path, tmp_path: Path):
    specimen = tmp_path / OWNED_NAME
    specimen.write_bytes(sample_pdf.read_bytes())
    result = ingest_pdf(specimen)
    yield result
    with get_pool().connection() as conn:
        conn.execute("DELETE FROM documents WHERE filename = %s", (OWNED_NAME,))


def test_the_demo_page_is_served_at_the_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<form" in response.text or "<input" in response.text


def test_the_page_talks_to_the_real_endpoints():
    """A page referencing endpoints that do not exist would render and silently fail."""
    body = client.get("/").text

    assert "/v1/ingest" in body
    assert "/v1/query" in body
    assert "/v1/documents" in body


def test_the_documents_endpoint_lists_what_has_been_ingested(ingested):
    response = client.get("/v1/documents")

    assert response.status_code == 200
    documents = response.json()
    mine = [d for d in documents if d["filename"] == OWNED_NAME]

    assert len(mine) == 1
    assert mine[0]["document_id"] == ingested.document_id
    assert mine[0]["page_count"] == ingested.page_count
    assert mine[0]["chunk_count"] == ingested.chunk_count


def test_a_document_with_no_chunks_would_still_be_listed():
    """Counting through a plain join would hide a document whose chunks failed to
    write - exactly the broken state worth seeing. The count must come from an outer
    join so a zero is reported rather than the row disappearing."""
    with get_pool().connection() as conn:
        conn.execute(
            "INSERT INTO documents (filename, page_count) VALUES (%s, 3)"
            " ON CONFLICT (filename) DO NOTHING",
            ("orphan-demo.pdf",),
        )
    try:
        listed = {d["filename"]: d for d in client.get("/v1/documents").json()}
        assert "orphan-demo.pdf" in listed
        assert listed["orphan-demo.pdf"]["chunk_count"] == 0
    finally:
        with get_pool().connection() as conn:
            conn.execute("DELETE FROM documents WHERE filename = %s", ("orphan-demo.pdf",))
