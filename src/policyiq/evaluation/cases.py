"""The golden set: questions with known answers, and where those answers are.

Relevance is recorded as (filename, page), not as chunk ids. Chunk ids change every time
chunking changes - the section-label work re-cut the whole corpus - and a golden set
keyed on them would silently stop matching anything. Pages are what citations show a
reader, and they only change if the document does.

Each answerable case also carries its `evidence`: a pattern matching only the clause
that answers it. The pages are found by that pattern, reviewed, and then frozen into
`sources`, so the file says both WHERE the answer is and HOW that was decided - and
`gold check` can tell when the two disagree.
"""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_QUESTIONS = Path("evaluation/questions.json")

Category = Literal["coverage", "limit", "waiting-period", "exclusion", "terms", "claims",
                   "definition", "product-specific", "out-of-scope"]


class Source(BaseModel):
    filename: str
    pages: list[int] = Field(min_length=1)


class Case(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    question: str = Field(min_length=5)
    category: Category
    expect: Literal["answer", "refusal"]

    # Where the answer is. Required for "answer", forbidden for "refusal" - a question
    # the corpus does not answer has nowhere to point.
    evidence: str | None = None
    # Filename substrings limiting where evidence counts, for questions about one
    # product. Without it, a question about Star Health would accept Arogya's clause.
    documents: list[str] = []
    sources: list[Source] = []

    # Checks on the model's answer, as case-insensitive patterns. Crude, and meant to be:
    # "does it say 36 months" is objective, where "is it a good answer" needs a judge
    # this project does not have (see docs: a 3B model cannot mark its own work).
    must_match: list[str] = []
    must_not_match: list[str] = []
    note: str = ""

    @field_validator("evidence", "must_match", "must_not_match")
    @classmethod
    def patterns_compile(cls, value):
        # A typo in a pattern would otherwise surface as a case that silently never
        # matches, which scores as a wrong answer rather than a broken test.
        for pattern in [value] if isinstance(value, str) else value or []:
            try:
                re.compile(pattern)
            except re.error as exc:
                # re.error is not a ValueError, so pydantic would let it escape as a
                # bare traceback instead of a validation error naming the field.
                raise ValueError(f"invalid pattern {pattern!r}: {exc}") from exc
        return value

    @model_validator(mode="after")
    def evidence_matches_expectation(self):
        if self.expect == "answer" and not self.evidence:
            raise ValueError(f"{self.id}: an answerable case needs evidence")
        if self.expect == "refusal" and (self.evidence or self.sources):
            raise ValueError(f"{self.id}: a refusal case cannot have sources")
        return self

    def gold_pages(self) -> set[tuple[str, int]]:
        return {(s.filename, p) for s in self.sources for p in s.pages}


def load_cases(path: Path = DEFAULT_QUESTIONS) -> list[Case]:
    cases = [Case.model_validate(raw) for raw in json.loads(path.read_text())]
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"duplicate case id: {case.id}")
        seen.add(case.id)
    return cases


def save_cases(cases: list[Case], path: Path = DEFAULT_QUESTIONS) -> None:
    raw = [c.model_dump(exclude_defaults=True) for c in cases]
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
