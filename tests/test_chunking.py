import re

from policyiq.ingest.chunking import chunk_pages


def test_chunks_carry_the_page_they_came_from():
    pages = [(1, "Alpha clause text. " * 40), (2, "Beta clause text. " * 40)]
    chunks = chunk_pages(pages, target_chars=200, overlap_chars=20)
    assert {c.page_number for c in chunks} == {1, 2}
    assert all(c.page_number in (1, 2) for c in chunks)


def test_chunk_index_is_globally_sequential():
    pages = [(1, "Alpha. " * 100), (2, "Beta. " * 100)]
    chunks = chunk_pages(pages, target_chars=200, overlap_chars=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_prefers_clause_boundaries_over_fixed_width():
    text = "1. Definitions.\n\n2. Coverage applies to hospitalisation.\n\n3. Exclusions apply."
    chunks = chunk_pages([(1, text)], target_chars=40, overlap_chars=0)
    assert len(chunks) >= 2
    assert not any(c.content.strip().endswith("hospitalis") for c in chunks)


def test_oversized_single_clause_is_split_rather_than_dropped():
    pages = [(1, "X" * 5000)]
    chunks = chunk_pages(pages, target_chars=500, overlap_chars=50)
    assert len(chunks) > 1
    assert sum(len(c.content) for c in chunks) >= 5000


def test_no_empty_chunks():
    pages = [(1, "\n\n\n   \n\n"), (2, "Real content here.")]
    chunks = chunk_pages(pages, target_chars=200, overlap_chars=20)
    assert all(c.content.strip() for c in chunks)


# --- Regression tests for bugs found in review, 2026-09-15 -------------------
# Each of these failed against the original implementation while the five tests
# above all passed. They assert properties rather than examples, which is what
# the original tests were missing.


def _pages_from(text: str) -> list[tuple[int, str]]:
    return [(1, text)]


def test_zero_overlap_does_not_repeat_previous_chunk_content():
    """BUG 1: current_chunk[-overlap_chars:] with overlap_chars == 0 evaluates to
    current_chunk[0:] — the whole string — so zero overlap produced total overlap."""
    text = "1. Definitions.\n\n2. Coverage applies to hospitalisation.\n\n3. Exclusions apply."
    chunks = chunk_pages(_pages_from(text), target_chars=40, overlap_chars=0)
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert earlier.content not in later.content, (
            f"chunk {later.chunk_index} contains all of chunk {earlier.chunk_index}"
        )


def test_no_chunk_materially_exceeds_the_target():
    """BUG 3: the hard-split loop only ran when current_chunk was empty, so a long
    clause arriving after a short one was never split.

    A chunk may legitimately exceed target_chars by up to the overlap it carries
    plus one separator character, but no further."""
    target, overlap = 100, 10
    pages = _pages_from("Short intro.\n\n" + "Y" * 600)
    chunks = chunk_pages(pages, target_chars=target, overlap_chars=overlap)
    ceiling = target + overlap + 1
    oversized = [(c.chunk_index, len(c.content)) for c in chunks if len(c.content) > ceiling]
    assert not oversized, f"chunks exceed {ceiling} chars: {oversized}"


def test_merged_clauses_are_separated_by_whitespace():
    """BUG 2: the split consumes the newlines between clauses, and merging with
    += welded them together — 'withdrawn.6. WAITING PERIOD'. That corrupts the
    text an embedding sees and makes cited excerpts look broken."""
    text = "1. Definitions.\n\n2. Coverage applies."
    content = chunk_pages(_pages_from(text), target_chars=500, overlap_chars=50)[0].content
    assert "Definitions.2." not in content
    assert re.search(r"Definitions\.\s+2\.", content), repr(content)


def test_real_corpus_respects_the_target(sample_pdf):
    """End-to-end guard against BUG 3 on actual documents, where 16% of chunks
    exceeded the target and the largest was 2.4x it."""
    from policyiq.ingest.pdf import extract_pages

    target, overlap = 1200, 150
    chunks = chunk_pages(extract_pages(sample_pdf), target, overlap)
    assert chunks
    ceiling = target + overlap + 1
    worst = max(len(c.content) for c in chunks)
    assert worst <= ceiling, f"largest chunk {worst} chars, ceiling {ceiling}"


def test_overlap_never_crosses_a_page_boundary():
    """Not a bug — locking in correct behaviour so a later refactor cannot break it.
    Carrying overlap across pages would attach one page's text to another page's
    number, silently corrupting every citation that followed."""
    pages = [(1, "Alpha content on page one. " * 20), (2, "Beta content on page two. " * 20)]
    chunks = chunk_pages(pages, target_chars=200, overlap_chars=80)
    for chunk in chunks:
        expected = "Alpha" if chunk.page_number == 1 else "Beta"
        forbidden = "Beta" if chunk.page_number == 1 else "Alpha"
        assert expected in chunk.content
        assert forbidden not in chunk.content, (
            f"page {chunk.page_number} chunk contains text from the other page"
        )
