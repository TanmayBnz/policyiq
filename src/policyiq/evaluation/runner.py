"""Running the golden set against the live system and summarising the result.

A report records the configuration it was produced under - commit, model, top-k, chunk
sizes, prompt fingerprint, corpus size - because a number without its configuration
cannot be compared with anything, and comparison is the whole point: "did this change
help, and did it break something else?"
"""

import hashlib
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field

import policyiq.answer as answer_module
from policyiq.answer import answer_question
from policyiq.config import settings
from policyiq.db import list_documents
from policyiq.evaluation.cases import Case
from policyiq.evaluation.scoring import score_answer, score_retrieval
from policyiq.llm import get_provider
from policyiq.retrieval.vector import vector_search


@dataclass
class CaseResult:
    id: str
    category: str
    expect: str
    question: str
    retrieved: list[dict]
    hit: bool | None = None
    reciprocal_rank: float | None = None
    precision: float | None = None
    answer: str | None = None
    passed: bool | None = None
    failures: list[str] = field(default_factory=list)
    citations: int = 0
    citations_on_gold: int = 0
    seconds: float = 0.0


def missing_documents(cases: list[Case]) -> list[str]:
    """Gold filenames that are not ingested.

    Checked before running, and fatal. A missing document makes every question about it
    a retrieval miss, and the report would blame search for a corpus that was never
    loaded.
    """
    ingested = {filename for _, filename, _, _ in list_documents()}
    wanted = {s.filename for c in cases for s in c.sources}
    return sorted(wanted - ingested)


def run_case(case: Case, top_k: int, generate: bool, cold: bool = True) -> CaseResult:
    started = time.perf_counter()
    chunks = vector_search(case.question, top_k)
    result = CaseResult(
        id=case.id, category=case.category, expect=case.expect, question=case.question,
        retrieved=[
            {"filename": c.filename, "page": c.page_number, "score": round(c.score, 4),
             "gold": (c.filename, c.page_number) in case.gold_pages()}
            for c in chunks
        ],
    )
    if case.expect == "answer":
        retrieval = score_retrieval(chunks, case)
        result.hit = retrieval.hit
        result.reciprocal_rank = retrieval.reciprocal_rank
        result.precision = retrieval.precision

    if generate:
        # answer_question searches again rather than taking these chunks. Search is
        # deterministic, so both see the same passages, and the harness exercises the
        # exact code path a user's request takes rather than a copy of it.
        if cold:
            # Every question from a freshly loaded model. Without it an answer depends on
            # the questions asked before it, so two runs of the same code disagree and a
            # change cannot be told apart from noise. See OllamaProvider.unload.
            unload = getattr(get_provider(), "unload", None)
            if unload is None:
                raise RuntimeError("cold runs need a provider that can unload its model")
            unload()
        response = answer_question(case.question, top_k)
        score = score_answer(response, case)
        result.answer = response.answer
        result.passed = score.passed
        result.failures = score.failures
        result.citations = score.citations
        result.citations_on_gold = score.citations_on_gold

    result.seconds = round(time.perf_counter() - started, 2)
    return result


def summarise(results: list[CaseResult]) -> dict:
    answerable = [r for r in results if r.expect == "answer"]
    n = len(answerable)
    summary: dict = {
        "cases": len(results),
        "retrieval": {
            "answerable_cases": n,
            "hit_at_k": round(sum(r.hit for r in answerable) / n, 3) if n else None,
            "mrr": round(sum(r.reciprocal_rank for r in answerable) / n, 3) if n else None,
            "precision_at_k": round(sum(r.precision for r in answerable) / n, 3) if n else None,
            "misses": [r.id for r in answerable if not r.hit],
        },
    }
    generated = [r for r in results if r.passed is not None]
    if generated:
        by_category: dict[str, list[bool]] = defaultdict(list)
        for r in generated:
            by_category[r.category].append(bool(r.passed))
        cited = sum(r.citations for r in answerable if r.passed is not None)
        summary["answers"] = {
            "pass_rate": round(sum(r.passed for r in generated) / len(generated), 3),
            "passed": sum(bool(r.passed) for r in generated),
            "of": len(generated),
            "by_category": {
                k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_category.items())
            },
            # Of all citations on answerable questions, the share pointing at a gold
            # page - the project's headline claim, as a number.
            "citation_precision": (
                round(sum(r.citations_on_gold for r in answerable) / cited, 3)
                if cited else None
            ),
            "refused_answerable": [
                r.id for r in generated if "refused an answerable question" in r.failures
            ],
            "failed": [r.id for r in generated if not r.passed],
        }
    return summary


def configuration(top_k: int, generate: bool, cold: bool) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", False
    documents = list_documents()
    return {
        "commit": commit + ("+uncommitted" if dirty else ""),
        "top_k": top_k,
        "generate": generate,
        # Warm runs are faster and not repeatable; never compare one with a cold run.
        "cold_model": cold if generate else None,
        "model": settings.ollama_model if generate else None,
        "embedding_model": settings.embedding_model,
        "chunk_target_chars": settings.chunk_target_chars,
        "chunk_overlap_chars": settings.chunk_overlap_chars,
        # A fingerprint rather than the text: enough to tell two runs apart, and the
        # prompt itself lives in the code at that commit.
        "prompt_sha256": hashlib.sha256(
            answer_module.PROMPT_TEMPLATE.encode()
        ).hexdigest()[:12],
        "documents": len(documents),
        "chunks": sum(chunks for _, _, _, chunks in documents),
    }


def run(
    cases: list[Case], top_k: int, generate: bool, cold: bool = True, progress=None
) -> dict:
    missing = missing_documents(cases)
    if missing:
        raise RuntimeError(f"gold documents not ingested: {missing}")

    config = configuration(top_k, generate, cold)
    started = time.perf_counter()
    results = []
    for case in cases:
        result = run_case(case, top_k, generate, cold)
        results.append(result)
        if progress:
            progress(result)
    return {
        "configuration": config,
        "summary": summarise(results),
        "seconds": round(time.perf_counter() - started, 1),
        "results": [r.__dict__ for r in results],
    }
