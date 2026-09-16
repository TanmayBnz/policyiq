"""Finding and checking the pages each answer lives on.

`build` runs each case's evidence pattern over the corpus and records the matching
pages. `check` runs it again and reports any case whose recorded pages no longer agree -
a re-extracted document, a changed furniture rule, or an edited pattern. Both need the
PDFs, which are not redistributable, so both are local-only; the frozen pages are what
make the rest of the harness runnable without them.
"""

import re
from pathlib import Path

from policyiq.evaluation.cases import Case, Source
from policyiq.ingest.pdf import extract_pages, strip_page_furniture

PagesByFile = dict[str, list[tuple[int, str]]]


def load_corpus_pages(corpus_dir: Path) -> PagesByFile:
    """Page text exactly as ingestion sees it, so page numbers agree with the database."""
    return {
        pdf.name: strip_page_furniture(extract_pages(pdf))
        for pdf in sorted(corpus_dir.glob("*.pdf"))
    }


def evidence_pages(case: Case, corpus: PagesByFile) -> list[Source]:
    """Pages whose text matches the case's evidence pattern.

    Whitespace is collapsed first. PDF extraction breaks lines mid-sentence, and a
    pattern written against the clause as a reader sees it would otherwise miss half
    its occurrences.
    """
    if not case.evidence:
        return []
    pattern = re.compile(case.evidence, re.IGNORECASE)
    sources = []
    for filename, pages in corpus.items():
        if case.documents and not any(d.lower() in filename.lower() for d in case.documents):
            continue
        matched = [n for n, text in pages if pattern.search(" ".join(text.split()))]
        if matched:
            sources.append(Source(filename=filename, pages=matched))
    return sources


def build(cases: list[Case], corpus: PagesByFile) -> list[Case]:
    built = []
    for case in cases:
        sources = evidence_pages(case, corpus)
        if case.expect == "answer" and not sources:
            # An answerable question with nowhere to point would score every retrieval
            # as a miss. That is a broken case, not a result.
            raise ValueError(f"{case.id}: evidence matches no page in the corpus")
        built.append(case.model_copy(update={"sources": sources}))
    return built


def check(cases: list[Case], corpus: PagesByFile) -> list[str]:
    problems = []
    for case in cases:
        found = {(s.filename, p) for s in evidence_pages(case, corpus) for p in s.pages}
        recorded = case.gold_pages()
        if found != recorded:
            problems.append(
                f"{case.id}: recorded pages differ from evidence -"
                f" missing {sorted(found - recorded)}, stale {sorted(recorded - found)}"
            )
    return problems
