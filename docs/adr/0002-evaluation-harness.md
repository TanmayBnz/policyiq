# ADR 0002: Evaluation against a golden set, scored without a model judge

## Status
Accepted — 2026-09-16

## Context
Without measurement, every change to retrieval, chunking or the prompt is a guess.
That was demonstrated directly: a prompt change that fixed one wrong answer broke two
questions that had been answered correctly, and it was only caught by chance.

The usual way to score answers is to have a strong model judge them. This project has
no budget for a hosted model and no document text may leave the machine, so the only
available judge is the same 3B model that produced the answers. A judge that weak
produces numbers that look rigorous and are not.

## Decision
A golden set of questions (`evaluation/questions.json`), each recording where its
answer is, scored by `python -m policyiq.evaluation run`.

**Relevance is recorded as (document, page), not chunk ids.** Chunk ids change every
time chunking changes; pages change only if the document does, and pages are what a
citation shows a reader.

**Each answer's pages are found by an evidence pattern, then frozen.** A case carries
a regular expression matching only the clause that answers it. `gold build` finds the
matching pages, which are reviewed and stored with the case; `gold check` reports any
case whose stored pages no longer match its pattern. The file records both where the
answer is and how that was decided.

**Retrieval and answers are scored separately.** A wrong answer from a missed clause
and a wrong answer from a misread clause have opposite fixes.

- Retrieval: hit@k, mean reciprocal rank, precision@k — set arithmetic, no judge.
- Answers: checked by patterns ("says 36 months", "says excluded", "does not say
  covered"), by whether a refusal was correct, and by whether at least one citation
  points at a gold page.
- Citation precision: the share of all citations that point at a gold page.

**Each question runs against a freshly loaded model.** Measured: at temperature 0, two
identical runs gave different answers to 17 of 33 questions, and 4 flipped between pass
and fail. Answers depended on what the model server had processed before them.
Reloading the model before every question made answers identical regardless of order,
at a cost of about 3.5 times the run time (667 s against 192 s for 35 questions). A `--warm` flag trades repeatability for
speed and is recorded in the report.

**Every report records its configuration** — commit, model, top-k, chunk sizes, a
prompt fingerprint, corpus size, cold or warm — because a number without its
configuration cannot be compared with anything.

## Consequences
- Pattern checks are crude. They can pass an answer that states the right number for
  the wrong reason, and fail a correct answer phrased unexpectedly. Every failure is
  listed by name in the report so it can be read, not only counted.
- The golden set targets documents that are not redistributable. Page lists are
  frozen in the file so the harness's own tests run without them; running the
  evaluation itself needs the corpus.
- A question expected to be unanswerable must be checked against the corpus text
  before it is trusted. One was not, and the model correctly answered it from an
  optional travel benefit.
- A golden set built from the most common answer marks a correct retrieval of a
  different product's answer as a miss. Questions whose answer differs by product
  either accept every product's figure or name the product.
