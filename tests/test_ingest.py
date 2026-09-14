"""Ingestion pipeline: PDF in, rows in Postgres out.

These tests hit a real database rather than a mocked one. The things most likely to
break here — the delete-then-insert replacement, the cascade, the vector column
accepting what the embedder produces — are all database behaviour, and a mock would
assert that the mock works.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from policyiq.db import get_pool
from policyiq.ingest.pipeline import ingest_pdf
from policyiq.main import app

client = TestClient(app)


OWNED_NAMES = ["ingest-specimen.pdf", "scanned.pdf", "evil.pdf"]


@pytest.fixture(scope="session")
def specimen_pdf(sample_pdf: Path, tmp_path_factory) -> Path:
    """The sample document's bytes under a filename these tests own.

    Ingestion is keyed on filename, so ingesting the corpus document under its real
    name would let a test delete or replace a row the working corpus depends on. A
    copy keeps the content real — the point of preferring the corpus — while making
    every row these tests create disposable.
    """
    target = tmp_path_factory.mktemp("ingest") / "ingest-specimen.pdf"
    target.write_bytes(sample_pdf.read_bytes())
    return target


@pytest.fixture(autouse=True)
def clean_corpus():
    """Remove only the rows these tests create, before and after.

    A bare DELETE over documents would be simpler, but this database also holds the
    working corpus; wiping it would leave the next retrieval experiment silently
    returning nothing. Clearing beforehand also stops residue from a previous run
    making the duplicate-detection assertions pass for the wrong reason.
    """
    def purge():
        with get_pool().connection() as conn:
            conn.execute("DELETE FROM documents WHERE filename = ANY(%s)", (OWNED_NAMES,))

    purge()
    yield
    purge()


def test_ingest_stores_one_chunk_row_per_chunk(specimen_pdf: Path):
    result = ingest_pdf(specimen_pdf)

    assert result.chunk_count > 0
    assert result.page_count > 0
    with get_pool().connection() as conn:
        stored = conn.execute(
            "SELECT count(*) FROM chunks WHERE document_id = %s", (result.document_id,)
        ).fetchone()[0]
    assert stored == result.chunk_count


def test_stored_chunks_keep_their_page_numbers_and_text(specimen_pdf: Path):
    """Page attribution is what makes a citation real, so it has to survive the write."""
    result = ingest_pdf(specimen_pdf)

    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT page_number, chunk_index, content FROM chunks"
            " WHERE document_id = %s ORDER BY chunk_index",
            (result.document_id,),
        ).fetchall()

    assert [r[1] for r in rows] == list(range(len(rows)))
    assert all(r[0] >= 1 for r in rows)
    assert all(r[2].strip() for r in rows)


def test_reingesting_the_same_file_replaces_rather_than_duplicates(specimen_pdf: Path):
    """Without this, every dev-loop re-run silently doubles the corpus and retrieval
    fills up with near-identical neighbours."""
    first = ingest_pdf(specimen_pdf)
    second = ingest_pdf(specimen_pdf)

    assert first.chunk_count == second.chunk_count
    with get_pool().connection() as conn:
        documents = conn.execute(
            "SELECT count(*) FROM documents WHERE filename = %s", (second.filename,)
        ).fetchone()[0]
        # Counted through the join rather than over the whole table, so a corpus
        # ingested outside the tests cannot mask a leaked first copy.
        chunks = conn.execute(
            "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE d.filename = %s",
            (second.filename,),
        ).fetchone()[0]

    assert documents == 1
    assert chunks == second.chunk_count


def test_a_pdf_with_no_extractable_text_is_rejected(tmp_path: Path):
    """A scanned policy yields no text. Storing a zero-chunk document would leave a
    row that can never be retrieved and looks like a successful ingest."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    blank = tmp_path / "scanned.pdf"
    pdf.output(str(blank))

    with pytest.raises(ValueError):
        ingest_pdf(blank)

    with get_pool().connection() as conn:
        documents = conn.execute(
            "SELECT count(*) FROM documents WHERE filename = 'scanned.pdf'"
        ).fetchone()[0]
    assert documents == 0


def test_a_document_failing_intake_validation_is_rejected(tmp_path: Path):
    """Validation runs before anything is written, so a bad document cannot reach the
    corpus and start competing for space in search results."""
    from fpdf import FPDF

    body = (
        "1. The Company shall indemnify the Insured Person for Medical Expenses "
        "incurred towards Hospitalisation. 2. The premium and claim procedure are "
        "set out in the Schedule to this Policy issued by the insurer."
    )
    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for _ in range(8):
        pdf.add_page()
        pdf.multi_cell(0, 6, body)
    repeated = tmp_path / "scanned.pdf"
    pdf.output(str(repeated))

    with pytest.raises(ValueError, match="duplicate_pages"):
        ingest_pdf(repeated)

    with get_pool().connection() as conn:
        documents = conn.execute(
            "SELECT count(*) FROM documents WHERE filename = 'scanned.pdf'"
        ).fetchone()[0]
    assert documents == 0


def test_upload_endpoint_ingests_and_reports_what_it_stored(specimen_pdf: Path):
    with specimen_pdf.open("rb") as handle:
        response = client.post(
            "/v1/ingest", files={"file": (specimen_pdf.name, handle, "application/pdf")}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == specimen_pdf.name
    assert body["chunk_count"] > 0
    with get_pool().connection() as conn:
        stored = conn.execute(
            "SELECT count(*) FROM chunks WHERE document_id = %s", (body["document_id"],)
        ).fetchone()[0]
    assert stored == body["chunk_count"]


def test_upload_endpoint_strips_directories_from_the_supplied_name(specimen_pdf: Path):
    """The uploaded name is client-controlled and is used to build a path. Only the
    final component may survive, or an upload can write outside its temp directory."""
    with specimen_pdf.open("rb") as handle:
        response = client.post(
            "/v1/ingest",
            files={"file": ("../../evil.pdf", handle, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["filename"] == "evil.pdf"


def test_upload_endpoint_refuses_a_non_pdf():
    response = client.post(
        "/v1/ingest", files={"file": ("notes.txt", b"not a policy", "text/plain")}
    )
    assert response.status_code == 400


def test_upload_endpoint_reports_an_unreadable_pdf_as_unprocessable(tmp_path: Path):
    """A scan is a client problem, not a server fault, so it must not surface as a 500."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    blank = tmp_path / "scanned.pdf"
    pdf.output(str(blank))

    with blank.open("rb") as handle:
        response = client.post(
            "/v1/ingest", files={"file": ("scanned.pdf", handle, "application/pdf")}
        )

    assert response.status_code == 422
