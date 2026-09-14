import re
from dataclasses import dataclass

CLAUSE_BOUNDARY = re.compile(r"\n\s*(?=\d+\.\s|\([a-z]\)\s|[A-Z][A-Z ]{3,}\n)|\n\s*\n")

# The split consumes the whitespace it matched on, so merged segments must be
# rejoined with something. Without this they weld together as "withdrawn.6." —
# see BUGFIX 2 below.
SEPARATOR = "\n"


@dataclass(frozen=True)
class Chunk:
    page_number: int
    chunk_index: int
    content: str


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """Return the trailing `overlap_chars` characters, or nothing if overlap is off.

    BUGFIX 1: the original wrote `current_chunk[-overlap_chars:]` inline. Python has
    no negative zero, so when overlap_chars == 0 that becomes `current_chunk[0:]` —
    the ENTIRE string. Zero overlap therefore produced total overlap, and every chunk
    contained all of its predecessors. Guarding the <= 0 case is the whole fix.
    """
    if overlap_chars <= 0:
        return ""
    return text[-overlap_chars:]


def chunk_pages(
    pages: list[tuple[int, str]], target_chars: int, overlap_chars: int
) -> list[Chunk]:
    """Break pages into chunks that respect clause boundaries.

    Insurance policies are structured as numbered clauses and lettered sub-clauses.
    Splitting mid-clause can separate an obligation from the exclusions qualifying it,
    which produces answers that state the opposite of the policy — so segments are cut
    on clause boundaries first and only hard-split when a single segment is too big
    on its own.

    Args:
        pages: (page_number, text) pairs, page numbers 1-based.
        target_chars: soft size target for a chunk.
        overlap_chars: trailing characters carried into the next chunk, so a clause
            split across a boundary stays retrievable from either side.

    Guarantees:
        - No chunk exceeds target_chars + overlap_chars + 1.
        - Overlap never crosses a page boundary, so page attribution stays exact.
        - chunk_index is sequential across the whole document, not per page.
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    for page_number, text in pages:
        # Reset per page. Carrying a partial chunk across pages would attach one
        # page's text to another page's number and silently corrupt every citation
        # after it.
        current_chunk = ""

        for raw_clause in CLAUSE_BOUNDARY.split(text):
            clause = raw_clause.strip()
            if not clause:
                continue

            # BUGFIX 3: this oversized-clause branch used to live in the `else` of
            # the size check, so it only ran when current_chunk was empty. A long
            # clause arriving after a short one skipped the hard split entirely and
            # was emitted whole — 610 chars against a 100 target in testing, and 16%
            # of real chunks over target with the largest at 2.4x. Handling it first,
            # unconditionally, is what fixes that.
            if len(clause) > target_chars:
                if current_chunk:
                    chunks.append(Chunk(page_number, chunk_index, current_chunk))
                    chunk_index += 1
                    current_chunk = ""

                # Sliding window: each window starts overlap_chars before the end of
                # the previous one. The stride was already correct in the original.
                while len(clause) > target_chars:
                    chunks.append(Chunk(page_number, chunk_index, clause[:target_chars].strip()))
                    chunk_index += 1
                    clause = clause[target_chars - overlap_chars :]

                current_chunk = clause.strip()
                continue

            # BUGFIX 2: merging used `current_chunk += clause` with no separator.
            # The regex consumes the newlines it splits on, so clauses were welded
            # into "1. Definitions.2. Coverage" — which degrades the embedding and
            # makes cited excerpts look broken to anyone reading them.
            candidate = f"{current_chunk}{SEPARATOR}{clause}" if current_chunk else clause

            if len(candidate) > target_chars and current_chunk:
                chunks.append(Chunk(page_number, chunk_index, current_chunk))
                chunk_index += 1
                tail = _overlap_tail(current_chunk, overlap_chars)
                current_chunk = f"{tail}{SEPARATOR}{clause}" if tail else clause
            else:
                current_chunk = candidate

        if current_chunk.strip():
            chunks.append(Chunk(page_number, chunk_index, current_chunk.strip()))
            chunk_index += 1

    return chunks
