"""Reject unusable documents before they reach the corpus.

A bad document is expensive to remove once ingested: its passages are already
competing for space in every set of search results, and nothing about the failure is
visible at the point it matters. Each rule below was written after a real document
got through the obvious check — "does this PDF contain any text?" — and caused a
problem downstream.
"""

import re
from collections import Counter
from dataclasses import dataclass

CONTROL_CHARACTERS = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f]")
WORD = re.compile(r"[A-Za-z]{2,}")
WHITESPACE = re.compile(r"\s+")

# Substrings, not whole words, so that "hospitalisation" counts towards "hospital"
# and "reinsurer" towards "insurer". These are the terms no health policy wording
# can avoid using repeatedly, whichever insurer wrote it.
DOMAIN_TERMS = ("policy", "insured", "hospital", "premium", "claim", "insurer")

# Measured over the nine accepted documents: 48 to 67 occurrences per thousand words.
# The document with the broken font map scored near zero, because its terms had been
# mangled into "3olicy" and "IQsured". Ten leaves a wide margin below the observed
# band while still catching a document whose vocabulary has been destroyed.
MIN_DOMAIN_TERMS_PER_1000 = 10.0

# Control characters do not appear at all in a correctly extracted page. One corpus
# document carried them on 94% of its pages. Any single page could in principle pick
# one up from an odd glyph, so this triggers on a pattern rather than one occurrence.
MAX_CONTROL_CHARACTER_PAGE_RATIO = 0.05

# One sourced PDF held the entire policy text on every page. Repetition at that level
# makes a cited page number meaningless and floods results with near-identical
# passages. Genuine documents repeat the odd page; they do not repeat a third of them.
MAX_DUPLICATE_PAGE_RATIO = 0.30


@dataclass(frozen=True)
class Rejection:
    rule: str
    detail: str


def _normalise(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip().lower()


def check_document(pages: list[tuple[int, str]]) -> list[Rejection]:
    """Return every reason this document should not be ingested, or an empty list.

    All rules run rather than returning at the first failure, because a document that
    trips several is worth seeing whole — it usually means the source is wrong rather
    than the extraction.
    """
    if not pages:
        return [Rejection("no_text", "no page yielded extractable text")]

    rejections: list[Rejection] = []
    texts = [text for _, text in pages]

    affected = sum(1 for text in texts if CONTROL_CHARACTERS.search(text))
    ratio = affected / len(texts)
    if ratio > MAX_CONTROL_CHARACTER_PAGE_RATIO:
        rejections.append(
            Rejection(
                "control_characters",
                f"{ratio:.0%} of pages contain control characters, which indicates a "
                f"broken embedded font map rather than readable text",
            )
        )

    combined = " ".join(texts).lower()
    word_count = len(WORD.findall(combined))
    hits = sum(combined.count(term) for term in DOMAIN_TERMS)
    density = hits / max(word_count, 1) * 1000
    if density < MIN_DOMAIN_TERMS_PER_1000:
        rejections.append(
            Rejection(
                "low_domain_vocabulary",
                f"{density:.1f} domain terms per 1000 words, below {MIN_DOMAIN_TERMS_PER_1000}"
                f" - not a health policy wording, or its text did not survive extraction",
            )
        )

    counts = Counter(_normalise(text) for text in texts)
    duplicates = sum(count - 1 for count in counts.values())
    duplicate_ratio = duplicates / len(texts)
    if duplicate_ratio > MAX_DUPLICATE_PAGE_RATIO:
        rejections.append(
            Rejection(
                "duplicate_pages",
                f"{duplicate_ratio:.0%} of pages repeat text found on another page",
            )
        )

    return rejections
