from pathlib import Path

from policyiq.ingest.pdf import extract_pages, strip_page_furniture


def _paged(bodies: list[str], header: str = "", footer: str = "") -> list[tuple[int, str]]:
    return [
        (n, "\n".join(part for part in (header, body, footer) if part))
        for n, body in enumerate(bodies, start=1)
    ]


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


def test_a_header_repeated_on_every_page_is_removed():
    """Insurer letterheads are printed on every page. Stored as if they were policy
    text they open dozens of chunks with identical wording, which drags a document's
    passages toward each other and away from what they actually say."""
    pages = _paged(
        [f"{n}. Clause text unique to this page about coverage." for n in range(1, 9)],
        header="HDFC ERGO General Insurance Company Limited",
    )
    stripped = strip_page_furniture(pages)

    assert not any("HDFC ERGO" in text for _, text in stripped)
    assert all("Clause text unique" in text for _, text in stripped)


def test_a_footer_whose_page_number_changes_is_still_removed():
    """'Page 3 of 45' differs on every page, so matching on exact text misses it."""
    pages = _paged(
        [f"{n}. Clause text unique to this page." for n in range(1, 9)],
        footer="Page 1 of 8",
    )
    pages = [(n, text.replace("Page 1 of 8", f"Page {n} of 8")) for n, text in pages]

    assert not any("Page" in text for _, text in strip_page_furniture(pages))


def test_text_appearing_on_only_a_few_pages_is_kept():
    """A heading on two of eight pages is content, not furniture."""
    bodies = [f"{n}. Clause text unique to this page." for n in range(1, 9)]
    bodies[2] = "EXCLUSIONS\n" + bodies[2]
    bodies[3] = "EXCLUSIONS\n" + bodies[3]

    stripped = strip_page_furniture(_paged(bodies))
    assert sum("EXCLUSIONS" in text for _, text in stripped) == 2


def test_repeated_text_in_the_middle_of_a_page_is_kept():
    """Only the edges of a page are furniture. A sentence repeated mid-page is the
    document genuinely saying the same thing twice, and removing it would delete
    policy language."""
    body = "\n".join(
        [f"Opening line {i}." for i in range(20)]
        + ["The Company shall not be liable for any claim arising from war."]
        + [f"Closing line {i}." for i in range(20)]
    )
    stripped = strip_page_furniture([(n, body) for n in range(1, 9)])

    assert len(stripped) == 8, "pages must survive, or the assertion below is vacuous"
    assert all("shall not be liable" in text for _, text in stripped)


def test_a_short_document_is_left_alone():
    """With three pages, 'appears on most pages' is not evidence of anything."""
    pages = _paged(["1. Only clause.", "2. Second clause.", "3. Third clause."], header="NOTICE")
    assert strip_page_furniture(pages) == pages


def test_a_page_reduced_to_nothing_is_dropped_without_renumbering_the_rest():
    """A page carrying only a letterhead has no content to cite."""
    pages = _paged(
        [f"{n}. Clause text unique to this page." for n in range(1, 9)],
        header="ACME Insurance Company Limited",
    )
    pages[4] = (5, "ACME Insurance Company Limited")

    stripped = strip_page_furniture(pages)
    assert [n for n, _ in stripped] == [1, 2, 3, 4, 6, 7, 8]


def test_stripping_furniture_from_the_real_corpus_keeps_most_of_the_text(sample_pdf: Path):
    """A rule this blunt could quietly gut a document, so bound how much it removes."""
    pages = extract_pages(sample_pdf)
    before = sum(len(text) for _, text in pages)
    after = sum(len(text) for _, text in strip_page_furniture(pages))
    assert 0.80 < after / before <= 1.0
