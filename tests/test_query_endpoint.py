"""Grounded answers and citation mapping.

Most of these drive a stub model rather than a real one. That is deliberate: the
property under test is that citations come from the database rather than from the
model's reply, and the only way to prove it is to make the model say something
demonstrably wrong and check the wrongness does not survive. A live model cannot be
asked to hallucinate a specific citation on demand.

Two tests at the bottom use a real model and are the only ones that skip.
"""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from policyiq.answer import answer_question, build_prompt
from policyiq.db import get_pool
from policyiq.ingest.pipeline import ingest_pdf
from policyiq.llm import get_provider
from policyiq.main import app
from policyiq.retrieval.vector import vector_search

client = TestClient(app)

OWNED_NAME = "query-specimen.pdf"
QUESTION = "what treatments are excluded"


class StubProvider:
    """A model whose reply is chosen by the test. Records the prompts it was given."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply

    def healthy(self) -> bool:
        return True


@pytest.fixture(scope="module", autouse=True)
def corpus(sample_pdf: Path, tmp_path_factory):
    """One known document, ingested under a name these tests own.

    The working corpus is left in place: these assertions hold whichever documents are
    present, and searching a realistic corpus is more meaningful than searching one
    document.
    """
    specimen = tmp_path_factory.mktemp("query") / OWNED_NAME
    specimen.write_bytes(sample_pdf.read_bytes())
    ingest_pdf(specimen)
    yield
    with get_pool().connection() as conn:
        conn.execute("DELETE FROM documents WHERE filename = %s", (OWNED_NAME,))


@pytest.fixture
def stub(monkeypatch):
    """Install a stub model. The factory returns the provider so tests can inspect it."""

    def install(reply: str) -> StubProvider:
        provider = StubProvider(reply)
        monkeypatch.setattr("policyiq.answer.get_provider", lambda: provider)
        return provider

    return install


# --- the prompt ---------------------------------------------------------------------


def test_every_chunk_is_labelled_with_a_numbered_marker():
    chunks = vector_search(QUESTION, top_k=3)
    prompt = build_prompt(QUESTION, chunks)

    assert len(chunks) == 3
    for n in range(1, 4):
        assert f"[{n}]" in prompt


def test_the_prompt_carries_the_question_and_the_retrieved_text():
    chunks = vector_search(QUESTION, top_k=2)
    prompt = build_prompt(QUESTION, chunks)

    assert QUESTION in prompt
    for chunk in chunks:
        assert chunk.content.strip()[:60] in prompt


def test_the_prompt_permits_refusing_to_answer():
    """A model will invent an answer rather than admit the text does not contain one,
    unless it is told that saying so is allowed. "Not specified in these documents" is
    a correct and valuable answer."""
    prompt = build_prompt(QUESTION, vector_search(QUESTION, top_k=2)).lower()

    assert "only" in prompt, "the model must be told to use only the supplied excerpts"
    assert any(word in prompt for word in ("do not", "does not", "cannot", "not contain"))


# --- citation mapping: the centrepiece ----------------------------------------------


def test_only_the_markers_the_model_cited_become_citations(stub):
    stub("The policy excludes cosmetic surgery [2].")

    result = answer_question(QUESTION, top_k=3)
    retrieved = vector_search(QUESTION, top_k=3)

    assert len(result.citations) == 1
    assert result.citations[0].chunk_index == retrieved[1].chunk_index
    assert result.citations[0].filename == retrieved[1].filename


def test_a_marker_the_model_invented_is_discarded(stub):
    """The whole point of the design. Three excerpts were supplied and the model cites
    a ninth. A system that parsed citations out of the reply would emit a citation to a
    document that was never retrieved - indistinguishable, to a reader, from a real one.
    """
    stub("War is excluded [9] and so is dental treatment [42].")

    result = answer_question(QUESTION, top_k=3)

    assert result.citations == [], "markers outside the supplied range must not resolve"


def test_a_mix_of_real_and_invented_markers_keeps_only_the_real_ones(stub):
    stub("Cosmetic surgery is excluded [1], and so is war [7].")

    result = answer_question(QUESTION, top_k=3)
    retrieved = vector_search(QUESTION, top_k=3)

    assert len(result.citations) == 1
    assert result.citations[0].chunk_index == retrieved[0].chunk_index


def test_citation_details_come_from_the_database_not_the_reply(stub):
    """The model names a file and a page in its prose. Neither may reach the citation."""
    stub("According to totally-made-up.pdf on page 999, war is excluded [1].")

    result = answer_question(QUESTION, top_k=3)
    citation = result.citations[0]

    assert citation.filename != "totally-made-up.pdf"
    assert citation.page_number != 999
    with get_pool().connection() as conn:
        row = conn.execute(
            "SELECT d.filename, c.page_number FROM chunks c"
            " JOIN documents d ON d.id = c.document_id"
            " WHERE c.document_id = %s AND c.chunk_index = %s",
            (citation.document_id, citation.chunk_index),
        ).fetchone()
    assert row == (citation.filename, citation.page_number)


def test_each_chunk_is_cited_at_most_once(stub):
    stub("Excluded [1] and also excluded [1] and again [1].")

    result = answer_question(QUESTION, top_k=3)

    assert len(result.citations) == 1


def test_an_answer_that_cites_nothing_is_still_returned(stub):
    """A refusal is a legitimate answer and must not be swallowed for lacking markers."""
    stub("The provided policy documents do not cover this.")

    result = answer_question(QUESTION, top_k=3)

    assert result.answer.strip()
    assert result.citations == []


def test_an_excerpt_never_begins_part_way_through_a_word(stub):
    """Chunks carry the previous chunk's trailing overlap, so a chunk's first characters
    are usually the tail of the preceding clause - often mid-word, such as "ut of or in
    relation to". Quoted verbatim to a user, that reads as broken and undermines the
    trust a citation exists to create. The stored chunk must keep its overlap, because
    that is what makes a clause spanning a boundary retrievable; only the quoted extract
    needs trimming. How that is done is a design choice.
    """
    stub("Excluded [1] and [2] and [3].")

    result = answer_question(QUESTION, top_k=3)
    by_index = {c.chunk_index: c.content for c in vector_search(QUESTION, top_k=3)}

    assert result.citations
    for citation in result.citations:
        source = by_index[citation.chunk_index]
        head = citation.excerpt[:40]
        start = source.find(head)

        assert start != -1, "the excerpt must be verbatim from the chunk it cites"
        # The real requirement is that the quote begins at a word boundary. Asserting
        # it merely starts with a capital would be a stricter proxy than the rule, and
        # would fail on clauses that legitimately open mid-sentence.
        assert start == 0 or source[start - 1].isspace(), (
            f"excerpt starts mid-word: {citation.excerpt[:60]!r}"
        )


def test_every_citation_corresponds_to_a_chunk_that_was_retrieved(stub):
    stub("Excluded [1] and [2] and [3].")

    result = answer_question(QUESTION, top_k=3)
    retrieved = {(c.document_id, c.chunk_index) for c in vector_search(QUESTION, top_k=3)}

    assert {(c.document_id, c.chunk_index) for c in result.citations} <= retrieved


# --- the endpoint -------------------------------------------------------------------


def test_the_endpoint_returns_an_answer_and_citations(stub):
    stub("Cosmetic surgery is excluded [1].")

    response = client.post("/v1/query", json={"question": QUESTION, "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].strip()
    assert len(body["citations"]) == 1
    assert body["citations"][0]["page_number"] >= 1
    assert body["citations"][0]["filename"].endswith(".pdf")


def test_a_blank_question_is_rejected():
    assert client.post("/v1/query", json={"question": "   "}).status_code == 422


def test_a_missing_question_is_rejected():
    assert client.post("/v1/query", json={"top_k": 3}).status_code == 422


def test_top_k_defaults_when_it_is_not_supplied(stub):
    provider = stub("Excluded [1].")

    response = client.post("/v1/query", json={"question": QUESTION})

    assert response.status_code == 200
    markers = re.findall(r"\[(\d+)\]", provider.prompts[0])
    assert markers, "the prompt should carry numbered excerpts"


# --- integration: a real model ------------------------------------------------------


@pytest.fixture(scope="module")
def live():
    if not get_provider().healthy():
        pytest.skip("no model server reachable")


def test_a_real_model_answers_from_the_documents(live):
    result = answer_question("what is the waiting period for pre-existing diseases", top_k=5)

    assert result.answer.strip()
    assert result.citations, "a question the corpus answers should produce a citation"


def test_a_real_model_says_so_when_the_documents_do_not_answer(live):
    """Grounding, end to end. The corpus is health insurance; nothing in it covers this,
    and inventing an answer is the failure this whole design exists to prevent."""
    result = answer_question(
        "what is the annual mileage limit for a commercial goods vehicle?", top_k=5
    )

    assert re.search(r"do(es)? not|no information|not (specified|covered|mention)",
                     result.answer, re.IGNORECASE), result.answer


REFUSAL = "The provided policy documents do not cover this."


@pytest.mark.xfail(
    strict=True,
    reason="exclusion items reach the model without their section heading",
)
def test_a_real_model_recognises_an_exclusion_as_an_exclusion(live):
    """The worst failure seen so far: asked this, the system said maternity and fertility
    treatment were covered, citing the very clauses that exclude them. After making
    generation deterministic it refuses instead - no longer reversed, still wrong,
    because the documents do answer it. The specimen policy also lists both under its
    exclusions, so the question has a correct answer even where the real corpus is
    absent.

    Strict xfail: when chunks carry their section, this passes, the run fails, and the
    marker comes off."""
    result = answer_question(
        "Does the policy cover maternity expenses or infertility treatments?", top_k=5
    )

    assert result.answer.strip() != REFUSAL, "the documents do address this"
    assert re.search(r"exclu|not (be )?(covered|payable)|shall not be liable",
                     result.answer, re.IGNORECASE), result.answer
    assert result.citations, result.answer
