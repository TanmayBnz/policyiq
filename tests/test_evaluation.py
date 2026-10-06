"""The evaluation harness.

A harness that scores wrongly is worse than none: it turns a guess into a number that
looks measured. So every scoring rule is tested here against hand-built results, with no
database or model involved, and the runner is tested with search and generation
replaced by fixed returns - deterministic, and runnable in CI.
"""

import json

import pytest

from policyiq.answer import REFUSAL, build_prompt
from policyiq.evaluation import gold, runner
from policyiq.evaluation.cases import DEFAULT_QUESTIONS, Case, Source, load_cases
from policyiq.evaluation.scoring import is_refusal, score_answer, score_retrieval
from policyiq.retrieval.vector import RetrievedChunk
from policyiq.schemas import Citation, QueryResponse

GOLD = [Source(filename="a.pdf", pages=[3, 4])]


def answerable(**overrides) -> Case:
    fields = dict(
        id="t",
        question="What is the waiting period?",
        category="waiting-period",
        expect="answer",
        evidence="36 months",
        sources=GOLD,
        must_match=["36 months"],
    )
    return Case(**(fields | overrides))


def refusal_case() -> Case:
    return Case(id="r", question="Is my car covered?", category="out-of-scope", expect="refusal")


def chunk(filename: str, page: int, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=page,
        document_id=1,
        filename=filename,
        page_number=page,
        chunk_index=page,
        content="text",
        score=score,
    )


def cite(filename: str, page: int) -> Citation:
    return Citation(
        document_id=1, filename=filename, page_number=page, chunk_index=page, excerpt="text"
    )


# --- retrieval --------------------------------------------------------------------


def test_retrieval_records_the_rank_of_every_gold_chunk():
    chunks = [chunk("b.pdf", 3), chunk("a.pdf", 4), chunk("a.pdf", 9), chunk("a.pdf", 3)]
    score = score_retrieval(chunks, answerable())

    assert score.relevant_ranks == [2, 4]
    assert score.hit
    assert score.reciprocal_rank == 0.5
    assert score.precision == 0.5


def test_the_right_page_in_the_wrong_document_is_not_a_hit():
    """Nine documents are variants of one product and share page layouts. Page 3 of
    another file is not the answer's page."""
    score = score_retrieval([chunk("b.pdf", 3), chunk("b.pdf", 4)], answerable())

    assert not score.hit
    assert score.reciprocal_rank == 0.0
    assert score.precision == 0.0


# --- refusals ---------------------------------------------------------------------


def test_a_refusal_with_markers_appended_is_still_a_refusal():
    """Seen from the real model: "...do not cover this. [1], [2], [3]"."""
    assert is_refusal(f"{REFUSAL} [1], [2], [3], [4], [5]")
    assert is_refusal(f"  {REFUSAL.upper()}")


def test_an_answer_mentioning_coverage_is_not_a_refusal():
    assert not is_refusal("The policy does not cover maternity expenses [1].")


def test_the_prompt_asks_for_exactly_the_refusal_the_harness_looks_for():
    assert REFUSAL in build_prompt("anything", [])


# --- answers ----------------------------------------------------------------------


def test_a_correct_cited_answer_passes():
    response = QueryResponse(answer="It is 36 months [1].", citations=[cite("a.pdf", 3)])
    score = score_answer(response, answerable())

    assert score.passed, score.failures
    assert (score.citations, score.citations_on_gold) == (1, 1)


def test_refusing_an_answerable_question_fails():
    score = score_answer(QueryResponse(answer=REFUSAL, citations=[]), answerable())

    assert score.failures == ["refused an answerable question"]


def test_an_answer_missing_the_required_fact_fails():
    response = QueryResponse(answer="It is 24 months [1].", citations=[cite("a.pdf", 3)])

    assert score_answer(response, answerable()).failures == ["missing /36 months/"]


def test_an_answer_saying_something_forbidden_fails():
    case = answerable(must_not_match=[r"\bis covered"])
    response = QueryResponse(
        answer="After 36 months it becomes payable [1].", citations=[cite("a.pdf", 3)]
    )

    assert score_answer(response, case).passed
    response.answer = "Maternity is covered after 36 months [1]."
    assert score_answer(response, case).failures == [r"says forbidden /\bis covered/"]


def test_a_right_answer_citing_the_wrong_page_fails():
    """The headline claim is a checkable source. A correct sentence pinned to a page
    that does not hold it has not delivered that."""
    response = QueryResponse(answer="It is 36 months [1].", citations=[cite("a.pdf", 9)])
    score = score_answer(response, answerable())

    assert score.failures == ["no citation on a gold page"]
    assert (score.citations, score.citations_on_gold) == (1, 0)


def test_a_right_answer_citing_nothing_fails():
    response = QueryResponse(answer="It is 36 months.", citations=[])

    assert score_answer(response, answerable()).failures == ["no citation on a gold page"]


def test_a_clean_refusal_passes_an_out_of_scope_question():
    assert score_answer(QueryResponse(answer=REFUSAL, citations=[]), refusal_case()).passed


def test_answering_an_out_of_scope_question_fails():
    response = QueryResponse(answer="Yes, cars are covered [1].", citations=[cite("a.pdf", 3)])

    assert score_answer(response, refusal_case()).failures == [
        "answered a question the documents do not cover"
    ]


def test_a_refusal_with_citations_attached_fails():
    response = QueryResponse(answer=REFUSAL, citations=[cite("a.pdf", 3)])

    assert score_answer(response, refusal_case()).failures == ["refusal carries citations"]


# --- the golden set itself --------------------------------------------------------


def test_an_answerable_case_must_say_where_its_answer_is():
    with pytest.raises(ValueError, match="needs evidence"):
        Case(id="x", question="Is it covered?", category="coverage", expect="answer")


def test_a_refusal_case_cannot_point_at_pages():
    with pytest.raises(ValueError, match="cannot have sources"):
        Case(
            id="x",
            question="Is my car covered?",
            category="out-of-scope",
            expect="refusal",
            sources=GOLD,
        )


def test_a_broken_pattern_is_rejected_when_loaded():
    """Otherwise it would match nothing and score as a wrong answer, not a broken test."""
    with pytest.raises(ValueError, match="invalid pattern"):
        answerable(must_match=["36 (months"])


def test_duplicate_case_ids_are_rejected(tmp_path):
    raw = [answerable().model_dump(), answerable().model_dump()]
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


def test_the_committed_golden_set_is_complete():
    """Runs in CI, where the PDFs are absent: the frozen pages are what make the set
    usable there, so every answerable case must carry them."""
    cases = load_cases(DEFAULT_QUESTIONS)
    answerable_cases = [c for c in cases if c.expect == "answer"]

    assert len(cases) >= 30
    assert any(c.expect == "refusal" for c in cases)
    assert all(c.sources for c in answerable_cases), [
        c.id for c in answerable_cases if not c.sources
    ]
    assert all(c.must_match for c in answerable_cases)


# --- finding pages ----------------------------------------------------------------


CORPUS = {
    "arogya.pdf": [(1, "Definitions only."), (8, "excluded until the expiry of\n36   months")],
    "star-health.pdf": [(27, "excluded until the expiry of 36 months")],
}


def test_evidence_is_matched_across_line_breaks():
    case = answerable(evidence="expiry of 36 months", sources=[])

    assert gold.evidence_pages(case, CORPUS) == [
        Source(filename="arogya.pdf", pages=[8]),
        Source(filename="star-health.pdf", pages=[27]),
    ]


def test_evidence_can_be_limited_to_one_product():
    case = answerable(evidence="expiry of 36 months", documents=["star-health"], sources=[])

    assert gold.evidence_pages(case, CORPUS) == [Source(filename="star-health.pdf", pages=[27])]


def test_building_fails_on_evidence_that_matches_nothing():
    with pytest.raises(ValueError, match="matches no page"):
        gold.build([answerable(evidence="48 months", sources=[])], CORPUS)


def test_check_reports_recorded_pages_that_drifted():
    case = answerable(
        evidence="expiry of 36 months", sources=[Source(filename="arogya.pdf", pages=[1])]
    )

    [problem] = gold.check([case], CORPUS)
    assert "('arogya.pdf', 8)" in problem and "('star-health.pdf', 27)" in problem
    assert "stale [('arogya.pdf', 1)]" in problem


# --- the runner -------------------------------------------------------------------


@pytest.fixture
def fixed_system(monkeypatch):
    """Search, generation and the corpus listing replaced by fixed returns."""
    retrieved = {
        "What is the waiting period?": [chunk("b.pdf", 1), chunk("a.pdf", 3)],
        "Is my car covered?": [chunk("b.pdf", 2)],
    }
    answers = {
        "What is the waiting period?": QueryResponse(
            answer="36 months [2].", citations=[cite("a.pdf", 3)]
        ),
        "Is my car covered?": QueryResponse(answer=REFUSAL, citations=[]),
    }
    monkeypatch.setattr(runner, "retrieve", lambda q, k: retrieved[q])
    monkeypatch.setattr(runner, "answer_question", lambda q, k: answers[q])
    monkeypatch.setattr(
        runner, "list_documents", lambda: [(1, "a.pdf", 10, 40), (2, "b.pdf", 5, 20)]
    )
    monkeypatch.setattr(runner, "get_provider", lambda: provider)
    return answers


class RecordingProvider:
    def __init__(self):
        self.calls: list[str] = []

    def unload(self):
        self.calls.append("unload")


provider = RecordingProvider()


def test_a_run_summarises_retrieval_and_answers(fixed_system):
    report = runner.run([answerable(), refusal_case()], top_k=2, generate=True)
    summary = report["summary"]

    assert summary["retrieval"] == {
        "answerable_cases": 1,
        "hit_at_k": 1.0,
        # One answerable case cannot support a confident rate; the interval says so.
        "hit_at_k_ci95": (0.207, 1.0),
        "mrr": 0.5,
        "precision_at_k": 0.5,
        "misses": [],
    }
    assert summary["answers"]["pass_rate"] == 1.0
    assert summary["answers"]["citation_precision"] == 1.0
    assert summary["answers"]["by_category"] == {"out-of-scope": "1/1", "waiting-period": "1/1"}
    assert report["configuration"]["documents"] == 2
    assert report["configuration"]["chunks"] == 60
    # Retrieval numbers are meaningless without knowing which retriever made them.
    assert report["configuration"]["retrieval_mode"] in {"vector", "hybrid"}


def test_a_refused_answerable_question_is_listed_by_name(fixed_system):
    fixed_system["What is the waiting period?"] = QueryResponse(answer=REFUSAL, citations=[])

    answers = runner.run([answerable()], top_k=2, generate=True)["summary"]["answers"]

    assert answers["refused_answerable"] == ["t"]
    assert answers["failed"] == ["t"]
    assert answers["citation_precision"] is None


def test_a_cold_run_reloads_the_model_before_every_answer(fixed_system, monkeypatch):
    """Measured: at temperature 0, answers still depended on the questions asked before
    them, until the model was reloaded before each one."""
    order = []
    monkeypatch.setattr(provider, "unload", lambda: order.append("unload"))
    real = runner.answer_question
    monkeypatch.setattr(
        runner, "answer_question", lambda q, k: order.append("answer") or real(q, k)
    )

    runner.run([answerable(), refusal_case()], top_k=2, generate=True)

    assert order == ["unload", "answer", "unload", "answer"]


def test_a_warm_run_leaves_the_model_loaded(fixed_system, monkeypatch):
    monkeypatch.setattr(provider, "unload", lambda: pytest.fail("must not unload"))

    report = runner.run([answerable()], top_k=2, generate=True, cold=False)

    assert report["configuration"]["cold_model"] is False


def test_a_cold_run_refuses_a_provider_that_cannot_unload(fixed_system, monkeypatch):
    monkeypatch.setattr(runner, "get_provider", lambda: object())

    with pytest.raises(RuntimeError, match="unload"):
        runner.run([answerable()], top_k=2, generate=True)


def test_retrieval_only_runs_no_generation(fixed_system, monkeypatch):
    def fail(*_):
        raise AssertionError("generation must not run")

    monkeypatch.setattr(runner, "answer_question", fail)
    report = runner.run([answerable()], top_k=2, generate=False)

    assert "answers" not in report["summary"]
    assert report["results"][0]["passed"] is None


def test_a_gold_document_that_is_not_ingested_stops_the_run(fixed_system):
    """Otherwise every question about it scores as a search miss, blaming retrieval for
    a corpus that was never loaded."""
    case = answerable(sources=[Source(filename="missing.pdf", pages=[1])])

    with pytest.raises(RuntimeError, match="missing.pdf"):
        runner.run([case], top_k=2, generate=False)


# --- tolerance for a larger model's typography -------------------------------------
# From the 2026-10-06 hosted run: right answers scored as failures because of how the
# model typed. Each case below is a real string from that run's saved answers.


def test_scorer_ignores_narrow_no_break_space_and_non_breaking_hyphen():
    case = answerable(must_match=[r"5 ?%", r"pre-existing"])
    answer = "Limited to 5 % of the sum insured for pre‑existing diseases."
    assert score_answer(QueryResponse(answer=answer, citations=[cite("a.pdf", 3)]), case).passed


def test_cited_markers_accepts_the_formats_a_larger_model_writes():
    from policyiq.answer import _cited_markers

    assert _cited_markers("Yes [1] and [3].", 5) == [1, 3]
    assert _cited_markers("Covered【2†L24-L28】.", 5) == [2]  # fullwidth + range
    assert _cited_markers("Covered [2†L24-L28][3†i].", 5) == [2, 3]
    # Unchanged behaviour: out-of-range and repeated markers are still dropped.
    assert _cited_markers("[9] [1] [1]", 5) == [1]
    assert _cited_markers("no markers at all", 5) == []


# --- confidence interval ------------------------------------------------------------


def test_wilson_interval_stays_inside_zero_and_one_and_is_not_degenerate():
    from policyiq.evaluation.scoring import wilson_interval

    lo, hi = wilson_interval(21, 35)
    assert 0.42 < lo < 0.46 and 0.72 < hi < 0.76  # 0.60, +/- about 0.16
    # Every case passing must not claim certainty; none passing must not go below zero.
    assert wilson_interval(35, 35)[1] == 1.0 and wilson_interval(35, 35)[0] < 0.95
    assert wilson_interval(0, 35)[0] == 0.0 and wilson_interval(0, 35)[1] > 0.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
    # More data narrows it: the whole reason to grow the set.
    assert (lambda r: r[1] - r[0])(wilson_interval(60, 100)) < (lambda r: r[1] - r[0])(
        wilson_interval(21, 35)
    )


# --- split ------------------------------------------------------------------------


def test_cases_default_to_dev_and_accept_held_out():
    assert answerable().split == "dev"
    assert answerable(split="held-out").split == "held-out"
    with pytest.raises(ValueError):
        answerable(split="test")
