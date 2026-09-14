from pathlib import Path

from pypdf import PdfReader


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text per page.

    Page numbers are 1-based to match what a human sees in a PDF reader, because they
    are surfaced to users as citations — an off-by-one here makes every citation in the
    system subtly wrong and is invisible until someone checks a source by hand.

    Pages with no extractable text are skipped; they are usually scanned images or
    blank separators. A document where *every* page is empty is a scan, and the
    ingestion pipeline raises on that case rather than storing nothing silently.
    """
    reader = PdfReader(pdf_path)
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((index, text))
    return pages
