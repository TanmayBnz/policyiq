"""Scoring, kept free of I/O so every rule here is unit-tested without a database.

Retrieval and answers are scored separately, on purpose. Scoring only the final answer
cannot tell "the right clause was never retrieved" from "the model misread the right
clause", and those have opposite fixes - one is search, the other is the prompt or the
model. The maternity failure was the second kind, and only looked like the first.
"""

import re
from dataclasses import dataclass, field

from policyiq.answer import REFUSAL
from policyiq.evaluation.cases import Case
from policyiq.retrieval.vector import RetrievedChunk
from policyiq.schemas import QueryResponse


@dataclass(frozen=True)
class RetrievalScore:
    k: int
    # 1-based positions of retrieved chunks that sit on a gold page.
    relevant_ranks: list[int]

    @property
    def hit(self) -> bool:
        """At least one gold page in the top k. The ceiling on everything downstream:
        a clause that is not retrieved cannot be answered from."""
        return bool(self.relevant_ranks)

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of the first gold chunk. Separates "first" from "fifth", which hit@k
        scores the same - and the model reads the first excerpt first."""
        return 1 / self.relevant_ranks[0] if self.relevant_ranks else 0.0

    @property
    def precision(self) -> float:
        """Share of the k chunks that are on a gold page. Low precision means the
        prompt is mostly noise, which a small model handles badly."""
        return len(self.relevant_ranks) / self.k if self.k else 0.0


def score_retrieval(chunks: list[RetrievedChunk], case: Case) -> RetrievalScore:
    gold = case.gold_pages()
    ranks = [
        rank for rank, chunk in enumerate(chunks, start=1)
        if (chunk.filename, chunk.page_number) in gold
    ]
    return RetrievalScore(k=len(chunks), relevant_ranks=ranks)


def is_refusal(answer: str) -> bool:
    """Starts with the sentence the prompt prescribes.

    `startswith`, not equality: the model has been seen to append markers to a refusal
    ("...do not cover this. [1], [2]"), and that is still a refusal - one carrying
    citations, which is its own failure, scored below.
    """
    return answer.strip().lower().startswith(REFUSAL.lower())


@dataclass
class AnswerScore:
    failures: list[str] = field(default_factory=list)
    citations: int = 0
    # Citations pointing at a gold page. Only meaningful for answerable cases.
    citations_on_gold: int = 0

    @property
    def passed(self) -> bool:
        return not self.failures


def score_answer(response: QueryResponse, case: Case) -> AnswerScore:
    answer = response.answer
    score = AnswerScore(citations=len(response.citations))
    gold = case.gold_pages()
    score.citations_on_gold = sum(
        (c.filename, c.page_number) in gold for c in response.citations
    )

    for pattern in case.must_not_match:
        if re.search(pattern, answer, re.IGNORECASE):
            score.failures.append(f"says forbidden /{pattern}/")

    if case.expect == "refusal":
        if not is_refusal(answer):
            score.failures.append("answered a question the documents do not cover")
        elif response.citations:
            # A refusal with sources attached reads as "I checked these and they do not
            # cover it", which is a claim about those pages the model never made.
            score.failures.append("refusal carries citations")
        return score

    if is_refusal(answer):
        score.failures.append("refused an answerable question")
        return score
    for pattern in case.must_match:
        if not re.search(pattern, answer, re.IGNORECASE):
            score.failures.append(f"missing /{pattern}/")
    if not score.citations_on_gold:
        # The project's headline claim is an answer with a real, checkable source. An
        # answer that is right but cites nothing - or cites the wrong page - has not
        # delivered it.
        score.failures.append("no citation on a gold page")
    return score
