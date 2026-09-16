import re
from dataclasses import dataclass

CLAUSE_BOUNDARY = re.compile(r"\n\s*(?=\d+\.\s|\([a-z]\)\s|[A-Z][A-Z ]{3,}\n)|\n\s*\n")

# The split consumes the whitespace it matched on, so merged segments must be
# rejoined with something. Without this they weld together as "withdrawn.6." —
# see BUGFIX 2 below.
SEPARATOR = "\n"

# SECTION CONTEXT (added 2026-09-16). Asked whether maternity was covered, the system
# said yes, citing the clauses that exclude it. The chunks were the right ones, but the
# "Exclusions" heading was 3 pages earlier (niva-bupa-reassure-2: heading page 14, item
# page 17), so nothing in the text the model saw said they were exclusions. Every chunk
# now opens with the section it sits in, e.g.
#     Section: Exclusions > Standard Exclusions
#
# Headings are recognised by what they SAY, not how they are numbered. A survey of the
# corpus ruled numbering out: Arogya's "1. Benign ENT disorders" is a list item laid out
# exactly like Niva's top-level "5. Exclusions", and "B. Balloon Sinuplasty" like
# "B. PREAMBLE". What does hold across insurers is that the sections which change what a
# clause MEANS - does it grant cover, remove it, or only define a word - are named with
# a small vocabulary. Singular "cover" is deliberately absent: it matched every row of
# the non-payable items table ("65 TROLLY COVER").
SECTION_WORDS = {
    "exclusion": "exclusion",
    "definition": "definition",
    "benefit": "benefit",
    "coverage": "coverage",
    "covers": "coverage",
    # "General Terms and Clauses" and "Specified Terms and Conditions" are the same
    # section under different names, so they share a family.
    "terms": "terms",
    "condition": "terms",
    "clause": "terms",
}

# Optional list marker: 5. / 5.1. / E / E.i. / C. / ii. / II.  Uppercase-only letters
# and lowercase-only roman numerals, so the marker cannot swallow the first word of a
# title ("Standard" is not the marker "S").
_MARKER = r"(?:\d+(?:\.\d+)*\.?|[A-Z](?:\.[ivx]+)?\.?|[ivx]+\.|[IVX]+\.)"
_HEADING_LINE = re.compile(
    rf"^\s*(?P<marker>{_MARKER}\s+)?(?P<title>[A-Z][A-Za-z/& ]*?)\s*(?P<stop>\.)?\s*:?\s*$"
)
# Short connecting words that stay lowercase in a title-case heading.
_CONNECTORS = {"and", "of", "the", "under", "for", "to", "in", "on", "by", "a", "an", "or"}
# A heading longer than this is almost certainly a sentence that happened to wrap.
HEADING_MAX_WORDS = 6
# Bounds the section line, so the size guarantee below can state its worst case.
SECTION_TITLE_MAX_CHARS = 60

# A digit standing in for a letter inside a word. Policy_Document_Arogya_Sanjeevani
# really does print its heading as "7. EXCLUS1ONS"; unrecognised, the exclusion items
# after it were labelled with the section before - COVERAGE. Only digits with a letter
# on both sides are touched, so clause numbers and codes are left alone.
_DIGIT_IN_WORD = re.compile(r"(?<=[A-Za-z])[10](?=[A-Za-z])")

# An item ending in an IRDAI standard exclusion code, e.g. "7.8. Excluded Providers:
# (Code-Excl11)". The regulator assigns these codes to exclusions only, so such a line
# proves the section is an exclusions section even when its heading was missed. Anchored
# to the end of the line on purpose: star-health-assure's benefits text says "Exclusion
# no.1, (Code-Excl 01), Exclusion ..." to REFER to exclusions, mid-sentence.
_EXCLUSION_CODE_ITEM = re.compile(r"\(?\s*Code\s*[-–:]?\s*Excl\s*\d+\s*\)?\s*:?\s*$")
_INFERRED_EXCLUSIONS = ("Exclusions", "exclusion")


def _section_family(word: str) -> str | None:
    word = word.lower()
    if word in SECTION_WORDS:
        return SECTION_WORDS[word]
    if word.endswith("s") and word[:-1] in SECTION_WORDS:
        return SECTION_WORDS[word[:-1]]
    return None


def _letter_for_digit(match: re.Match[str]) -> str:
    return "I" if match.group() == "1" else "O"


def _heading(line: str) -> tuple[str, str] | None:
    """(title, family) if the line is a section heading, else None.

    Each condition below removed a class of false positive found in the corpus survey:
    - whole line, at most six words, no closing full stop (one exception, below):
      wrapped sentence fragments such as "and accepted without a specific exclusion.";
    - the section word first or last: "Insured Person has continuous coverage for";
    - all capitals, title case, or numbered and led by the section word:
      "Aplastic anaemia is a serious condition";
    - no " - ": list items like "ii. Chronic condition - A chronic condition".
    - last word of three letters or more: lines the PDF reader cut short, such as
      "3.9. Condition Pr", which otherwise relabelled the definitions after it.
    Starting with a capital also drops text the PDF reader damaged - niva-bupa-
    reassure-3 yields "tandard Exclusions" - which is a known gap, not a fix.
    """
    match = _HEADING_LINE.match(_DIGIT_IN_WORD.sub(_letter_for_digit, line))
    if not match:
        return None
    title = " ".join(match.group("title").split())
    words = title.split()
    if not words or len(words) > HEADING_MAX_WORDS or len(words[-1]) < 3:
        return None

    first, last = _section_family(words[0]), _section_family(words[-1])
    family = first or last
    if family is None:
        return None

    # A list marker vouches for a sentence-case title only when the title LEADS with the
    # section word ("D Benefits Covered under the policy"). A marked sentence ending in
    # one ("iii. The waiting period for listed conditions", star-health-assure) replaced
    # the exclusions label for two pages.
    title_case = all(w[0].isupper() or w.lower() in _CONNECTORS for w in words)
    marked_and_led = bool(match.group("marker") and first)
    if not (title.isupper() or title_case or marked_and_led):
        return None
    # A closing full stop usually means a wrapped sentence, except in the narrowest case:
    # niva-bupa-reassure-2 heads its benefits "4. Benefits available under the policy."
    # Missing it labelled eight pages of benefits as definitions.
    if match.group("stop") and not marked_and_led:
        return None
    return title[:SECTION_TITLE_MAX_CHARS], family


def _apply_heading(path: list[tuple[str, str]], heading: tuple[str, str]) -> list[tuple[str, str]]:
    """The section path after passing `heading`. Two levels at most.

    A bare section word ("Exclusions", "E Exclusion", "C. EXCLUSIONS") starts a new
    top-level section. A qualified one ("Standard Exclusions") nests under the current
    top level when both are the same kind of section, and otherwise starts its own - so
    "Specific Exclusions" replaces "Standard Exclusions" under "Exclusions", while
    "General Terms and Clauses" ends the exclusions entirely.
    """
    title, family = heading
    if len(title.split()) > 1 and path and path[0][1] == family:
        return [path[0], heading]
    return [heading]


def _section_at(text: str, offset: int, path: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The section in effect for a chunk whose own text starts at text[offset].

    Headings on complete lines before `offset` apply. So do headings the chunk itself
    opens with, since "5. Exclusions" at the top of a chunk describes that chunk. A
    heading at the END of a chunk is not seen here for that chunk - only for the next -
    because two of the three real layouts close the standard exclusion list with the
    "Specific Exclusions" heading, and labelling the list with it would misstate it.

    Only whole lines count. A hard split can cut a line in half, and half of
    "Standard Exclusions apply to..." would otherwise read as a heading.
    """
    leading = True
    for line in re.finditer(r"^.*$", text, re.MULTILINE):
        if line.end() <= offset:
            found = _heading(line.group())
            path = _apply_exclusion_code(path, line.group())
        elif line.start() >= offset and leading:
            found = _heading(line.group())
            if found is None and line.group().strip():
                # The chunk's first line of text describes the chunk, so a coded
                # exclusion item there relabels it.
                path = _apply_exclusion_code(path, line.group())
                leading = False
        else:
            break
        if found:
            path = _apply_heading(path, found)
    return path


def _apply_exclusion_code(path: list[tuple[str, str]], line: str) -> list[tuple[str, str]]:
    """Fall back to an exclusions section when a coded exclusion item turns up outside
    one. A missed heading otherwise leaves the PREVIOUS section's label in place, and a
    wrong label is worse than none: it states the reversal the label exists to prevent.

    Applied to lines before the chunk and to the chunk's first line, not to a coded
    item further down: a chunk that begins in the benefits and runs into the exclusions
    keeps its benefits label, and the next chunk carries the exclusions one.
    """
    if path and path[0][1] == "exclusion":
        return path
    if _EXCLUSION_CODE_ITEM.search(line):
        return [_INFERRED_EXCLUSIONS]
    return path


# Public: answer.py strips this line back out of quoted excerpts.
SECTION_PREFIX = "Section: "


def _section_line(path: list[tuple[str, str]]) -> str:
    return SECTION_PREFIX + " > ".join(title for title, _ in path) if path else ""


@dataclass(frozen=True)
class Chunk:
    page_number: int
    chunk_index: int
    content: str
    # The section line alone, also present as the first line of `content`. Kept
    # separately so the size guarantee can be checked against the body.
    section: str = ""


def _make_chunk(page_number: int, chunk_index: int, body: str, section: str) -> Chunk:
    # The section goes into `content` rather than only a separate column because
    # `content` is what gets embedded and what the model reads - the two places the
    # missing heading did its damage.
    content = f"{section}{SEPARATOR}{body}" if section else body
    return Chunk(page_number, chunk_index, content, section)


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
        - No chunk's body exceeds target_chars + overlap_chars + 1. The section line
          is extra: it is outside the budget, and at most two titles of
          SECTION_TITLE_MAX_CHARS each. Counting it would make the budget shift every
          time a heading did, for about 60 characters against a 512-token model limit.
        - Overlap never crosses a page boundary, so page attribution stays exact.
        - chunk_index is sequential across the whole document, not per page.
        - Every chunk names the section it sits in, once a heading has been seen.
    """
    chunks: list[Chunk] = []
    chunk_index = 0
    # NOT reset per page, unlike current_chunk below. The heading is three pages before
    # the item it governs, so the section has to survive page breaks. That does not
    # weaken the page rule: what crosses is a label naming the section, never any text
    # from the page the heading is on, so a citation still points at the only page the
    # chunk's text came from.
    path: list[tuple[str, str]] = []

    for page_number, text in pages:
        # Reset per page. Carrying a partial chunk across pages would attach one
        # page's text to another page's number and silently corrupt every citation
        # after it.
        current_chunk = ""
        chunk_section = ""

        for raw_clause in CLAUSE_BOUNDARY.split(text):
            clause = raw_clause.strip()
            if not clause:
                continue

            # The section for a chunk that starts with this clause, and the section
            # once the whole clause has been read. They differ when the clause has a
            # heading part-way through or at its end.
            opening_section = _section_line(_section_at(clause, 0, path))
            path_after = _section_at(clause, len(clause), path)

            # BUGFIX 3: this oversized-clause branch used to live in the `else` of
            # the size check, so it only ran when current_chunk was empty. A long
            # clause arriving after a short one skipped the hard split entirely and
            # was emitted whole — 610 chars against a 100 target in testing, and 16%
            # of real chunks over target with the largest at 2.4x. Handling it first,
            # unconditionally, is what fixes that.
            if len(clause) > target_chars:
                if current_chunk:
                    chunks.append(
                        _make_chunk(page_number, chunk_index, current_chunk, chunk_section)
                    )
                    chunk_index += 1
                    current_chunk = ""

                # Sliding window: each window starts overlap_chars before the end of
                # the previous one. The stride was already correct in the original.
                #
                # `offset` tracks where the window starts in the whole clause, so each
                # window gets the section in effect at ITS start. A long clause can
                # contain a heading, and windows after it belong to the new section.
                whole, offset = clause, 0
                while len(clause) > target_chars:
                    section = _section_line(_section_at(whole, offset, path))
                    window = clause[:target_chars].strip()
                    chunks.append(_make_chunk(page_number, chunk_index, window, section))
                    chunk_index += 1
                    clause = clause[target_chars - overlap_chars :]
                    offset += target_chars - overlap_chars

                current_chunk = clause.strip()
                chunk_section = _section_line(_section_at(whole, offset, path))
                path = path_after
                continue

            # BUGFIX 2: merging used `current_chunk += clause` with no separator.
            # The regex consumes the newlines it splits on, so clauses were welded
            # into "1. Definitions.2. Coverage" — which degrades the embedding and
            # makes cited excerpts look broken to anyone reading them.
            candidate = f"{current_chunk}{SEPARATOR}{clause}" if current_chunk else clause

            if len(candidate) > target_chars and current_chunk:
                chunks.append(_make_chunk(page_number, chunk_index, current_chunk, chunk_section))
                chunk_index += 1
                tail = _overlap_tail(current_chunk, overlap_chars)
                current_chunk = f"{tail}{SEPARATOR}{clause}" if tail else clause
                # The new chunk opens with the previous chunk's tail, which may belong
                # to the old section, but the clause is its new content - label it by
                # the clause. Using `path` as it stood would mislabel a chunk that
                # opens with "5. Exclusions" as part of whatever came before.
                chunk_section = opening_section
            else:
                if not current_chunk:
                    chunk_section = opening_section
                current_chunk = candidate

            # Only now, after this clause is placed. A heading at the end of this clause
            # governs the next chunk, not the one the clause just joined.
            path = path_after

        if current_chunk.strip():
            body = current_chunk.strip()
            chunks.append(_make_chunk(page_number, chunk_index, body, chunk_section))
            chunk_index += 1

    return chunks
