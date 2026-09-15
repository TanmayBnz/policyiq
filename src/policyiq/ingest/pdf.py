import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

# Only lines at the top or bottom of a page can be furniture. A sentence repeated in
# the middle of a page is the document genuinely saying the same thing twice, and
# removing it would delete policy language.
#
# Counted in non-blank lines, and set from what real letterheads actually look like:
# the largest in this corpus runs to fourteen lines of company name, registration
# number, two postal addresses, product code and contact details. Narrower windows
# were tried and caught only part of it, which leaves the remainder opening every
# chunk - worse than not stripping at all. The window is deliberately generous: it
# only nominates candidates, and the page-ratio threshold below is what actually
# decides, since genuine policy text does not repeat across half a document.
FURNITURE_EDGE_LINES = 15

# A line has to appear on at least half a document's pages before repetition is
# evidence of furniture rather than coincidence.
FURNITURE_PAGE_RATIO = 0.5

# Capping the length keeps a long clause that happens to recur from being mistaken
# for a header. The cap has to clear a real registered-office line, which runs to 133
# characters in this corpus. Length is the weak guard here; the page-ratio threshold
# below is what actually distinguishes furniture from content.
FURNITURE_MAX_CHARS = 200

# Below this, "appears on most pages" carries no information: on three pages, two
# occurrences is already 67%.
FURNITURE_MIN_PAGES = 5

# A page marker sits at the END of a line: "Page 7 of 55", "- 12 -", a bare number.
# Only that trailing part is removed before comparing lines, never digits generally:
# one insurer prints its letterhead and the page number on a single line, so leaving
# the number in makes every page's header unique and the repetition invisible - while
# collapsing digits everywhere would fuse "1. Waiting Periods" with "2. Exclusions"
# and strip numbered policy text as though it were furniture.
_PAGE_MARKER = re.compile(r"[-\s]*(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?[-\s]*$", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def _fingerprint(line: str) -> str:
    """Normalise a line so repetitions of it can be counted across pages."""
    normalised = _WHITESPACE.sub(" ", line).strip().lower()
    return _PAGE_MARKER.sub("", normalised).strip()


def strip_page_furniture(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Remove headers and footers that repeat across a document's pages.

    Insurers print a letterhead on every page. Stored as if it were policy text it
    opens dozens of chunks with identical wording, which pulls a document's passages
    toward each other and away from what they actually say, and produces a handful of
    chunks that are nothing but a letterhead and answer nothing.

    A page left with no content is dropped rather than stored empty. Remaining page
    numbers are untouched, because they are what citations point at.
    """
    if len(pages) < FURNITURE_MIN_PAGES:
        return pages

    def edges(lines: list[str]) -> set[int]:
        """Indices of the first and last few non-blank lines.

        Measured in non-blank lines because extraction leaves blank lines between the
        parts of a letterhead, and counting those spends the window on nothing.
        """
        filled = [i for i, line in enumerate(lines) if line.strip()]
        return set(filled[:FURNITURE_EDGE_LINES]) | set(filled[-FURNITURE_EDGE_LINES:])

    seen: Counter[str] = Counter()
    for _, text in pages:
        lines = text.splitlines()
        # Counted once per page, so a header appearing twice on one page cannot on its
        # own push a line over the threshold.
        seen.update(
            {
                _fingerprint(lines[i])
                for i in edges(lines)
                if lines[i].strip() and len(lines[i]) <= FURNITURE_MAX_CHARS
            }
        )

    threshold = len(pages) * FURNITURE_PAGE_RATIO
    furniture = {line for line, count in seen.items() if count >= threshold}
    if not furniture:
        return pages

    cleaned: list[tuple[int, str]] = []
    for page_number, text in pages:
        lines = text.splitlines()
        edge_indices = edges(lines)
        kept = [
            line
            for i, line in enumerate(lines)
            if not (i in edge_indices and _fingerprint(line) in furniture)
        ]
        body = "\n".join(kept).strip()
        if body:
            cleaned.append((page_number, body))
    return cleaned


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
