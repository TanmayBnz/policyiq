from pathlib import Path

from policyiq.ingest.pdf import extract_pages


def test_extract_pages_returns_one_based_page_numbers(sample_pdf: Path):
    pages = extract_pages(sample_pdf)
    assert pages, "expected at least one page with text"
    assert pages[0][0] == 1
    numbers = [n for n, _ in pages]
    assert numbers == sorted(numbers)
    assert len(set(numbers)) == len(numbers)


def test_extract_pages_skips_empty_pages_and_returns_real_text(sample_pdf: Path):
    pages = extract_pages(sample_pdf)
    assert all(text.strip() for _, text in pages)
    assert sum(len(t) for _, t in pages) > 2000


def test_page_numbers_survive_a_document_with_blank_pages(tmp_path: Path):
    """A blank page must not shift subsequent page numbers, or every citation after
    it points at the wrong page."""
    from pypdf import PdfReader, PdfWriter

    from tests.fixtures import write_synthetic_policy

    source = write_synthetic_policy(tmp_path / "src.pdf")
    reader, writer = PdfReader(source), PdfWriter()
    writer.add_page(reader.pages[0])
    writer.add_blank_page(width=595, height=842)
    writer.add_page(reader.pages[1])
    out = tmp_path / "with-blank.pdf"
    with out.open("wb") as fh:
        writer.write(fh)

    pages = extract_pages(out)
    assert [n for n, _ in pages] == [1, 3], "blank page 2 skipped, page 3 keeps its number"
