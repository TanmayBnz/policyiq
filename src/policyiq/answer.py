import re

from policyiq.config import settings
from policyiq.ingest.chunking import SECTION_PREFIX
from policyiq.llm import get_provider
from policyiq.retrieval.search import retrieve
from policyiq.retrieval.vector import RetrievedChunk
from policyiq.schemas import Citation, QueryResponse

# Long enough for a clause to be recognisable, short enough that a reader checks the
# page rather than reading the quote instead of the source.
EXCERPT_CHARS = 300

# The exact sentence the model is told to give when the excerpts do not answer. Named
# so the evaluation harness recognises a refusal by the same string the prompt asks for,
# rather than by a copy that could drift from it.
REFUSAL = "The provided policy documents do not cover this."

PROMPT_TEMPLATE = """You are answering questions about an insurance policy.

Use ONLY the numbered excerpts below. If they do not contain the answer, say
"{refusal}" Do not use outside knowledge.

Cite the excerpts you used with their bracketed numbers, for example [1] or [2].
Cite only numbers that appear below.

{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[{i}] {chunk.content.strip()}" for i, chunk in enumerate(chunks, start=1)
    )
    return PROMPT_TEMPLATE.format(context=context, question=question, refusal=REFUSAL)


# BUGFIX 2: excerpts were taken as `content.strip()[:300]`, starting at character 0.
# Every chunk after the first on a page opens with the previous chunk's 150-character
# overlap tail, so that slice usually began mid-word - real examples were
# "ng genetically modified organisms" and "es for treatment directly arising from".
# Quoted to a user beneath an answer, that reads as broken and undermines the trust a
# citation exists to create, even when the retrieval behind it was correct.
#
# The stored chunk keeps its overlap: that is what makes a clause spanning a boundary
# retrievable from either side. Only the quoted extract is trimmed, and only for
# display.
_EXCERPT_START = re.compile(r"(?:^|[.;:!?]\s+|\n\s*)([A-Z0-9(])")


def _excerpt(content: str, limit: int) -> str:
    """Quote from the first clean sentence start, not from the overlap tail."""
    text = content.strip()
    # BUGFIX (2026-09-16): chunks now open with a "Section: ..." line added by the
    # chunker. Left in, it was quoted as though it were the policy's own words, and -
    # since it sits at the very start - the search below matched it at position 0 and
    # never skipped the overlap tail that follows it. Dropping the line first restores
    # both: the quote is the document's text, and it starts at a clean sentence.
    if text.startswith(SECTION_PREFIX):
        text = text.partition("\n")[2].strip()
    match = _EXCERPT_START.search(text)
    if match:
        text = text[match.start(1) :]
    else:
        # No capitalised sentence start anywhere. Fall back to the next whole word, so
        # the quote at least does not begin part way through one.
        _, _, remainder = text.partition(" ")
        text = remainder or text
    return text[:limit].strip()


def _cited_markers(answer: str, maximum: int) -> list[int]:
    """Extract markers the model actually used, deduplicated, in order of appearance."""
    seen: list[int] = []
    for match in re.findall(r"\[(\d+)\]", answer):
        n = int(match)
        if 1 <= n <= maximum and n not in seen:
            seen.append(n)
    return seen


def answer_question(question: str, top_k: int | None = None) -> QueryResponse:
    k = top_k or settings.retrieval_top_k
    chunks = retrieve(question, k)
    if not chunks:
        return QueryResponse(
            answer="No documents have been ingested yet.", citations=[]
        )

    answer = get_provider().generate(build_prompt(question, chunks)).strip()

    # Citation metadata comes from the database, never from the model. The model
    # only selects WHICH retrieved chunks it used; it never supplies page numbers.
    markers = _cited_markers(answer, len(chunks))

    # BUGFIX 1: this was `used = [...] if markers else chunks`. The fallback meant that
    # when no valid marker was found, every retrieved chunk was cited - which inverted
    # the guarantee this function exists to provide, in exactly the two cases that
    # matter most:
    #
    #   "War is excluded [9] and dental [42]."           -> 3 citations
    #   "The provided policy documents do not cover this." -> 3 citations
    #
    # A model that cited nothing real got maximum apparent grounding, and a refusal
    # arrived with three sources attached, reading as "I checked these and they do not
    # cover it" when it actually meant "I cite everything I was handed".
    #
    # Citations mean *this answer came from here*. An answer that used nothing cites
    # nothing. Showing what was retrieved regardless is a reasonable thing to want, but
    # it is a different field with a different name, not this one.
    used = [chunks[n - 1] for n in markers]
    citations = [
        Citation(
            document_id=c.document_id,
            filename=c.filename,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            excerpt=_excerpt(c.content, EXCERPT_CHARS),
        )
        for c in used
    ]
    return QueryResponse(answer=answer, citations=citations)
