# ADR 0003: Hybrid search, fused by rank

## Status
Accepted — 2026-09-17

## Context
Vector search found an answer page in the top 5 for 27 of the 31 answerable golden
questions. The four misses had something in common. Three name a product or a date:
"HDFC ERGO Optima Secure", "Niva Bupa ReAssure", "policy wordings effective 12th
December 2022". An embedding represents what a passage is about, so a product name
barely moves it, and in one case the date appears only in the document's filename,
which vector search never sees. The fourth, "Is there a co-payment...", is spread across
nine near-identical documents.

These are the questions keyword matching is good at, and paraphrase is what it is bad
at, so the two were combined rather than one replacing the other.

## Decision
**Keyword search is Postgres full-text search** (`src/policyiq/retrieval/keyword.py`),
over a generated column (`sql/002_keyword_search.sql`). No second search engine: the
database already runs, and a generated column needs no change to ingestion and no
re-ingest.

**It matches any of the question's words, not all of them.** Postgres's own
`websearch_to_tsquery` requires every word, which suits a search box and fails on full
questions: 7 of 31 hits. Matching any word and ranking the results: 24 of 31.

**The filename's words are searchable with every chunk.** Measured, this was the largest
single gain: fused search went from 28 to 30 of 31 when it was added. It is also the
decision most specific to this corpus - see Consequences.

**Rankings are fused with reciprocal rank fusion** (`fusion.py`): each chunk scores
the sum of 1/(60 + rank) across the two rankings. Positions are combined, not scores,
because the two scores are on unrelated scales (cosine similarity around 0.5–0.9,
`ts_rank` around 0.01–0.1) and adding them lets the wider scale decide. Each retriever
contributes 20 candidates, and the fused list is cut to 5.

**Vector search stays the default; hybrid is opt-in** (`RETRIEVAL_MODE=hybrid`). This
was not the plan. Hybrid measured better at retrieval and worse at answers (below), and
the answer is what a user sees. The evaluation report records the mode, so the
comparison can be rerun once refusals are fixed.

## Measured
Golden set, top 5, 1,570 chunks:

| | Vector | Keyword alone | Hybrid |
|---|---|---|---|
| Answer page in top 5 | 27 / 31 | 28 / 31 | **30 / 31** |
| Mean reciprocal rank | 0.739 | 0.689 | **0.769** |
| Precision at 5 | 0.523 | | **0.587** |

Rank of the first answer page changed on 9 questions. Six improved, including all three
product/date misses and co-payment. Three got worse: `ambulance-limit` (1st to 3rd),
`day-care` and `niva-reassure-forever` (1st to 4th). They are still in the top 5, but
the 3B model reads the passages in order, so a lower rank is a real cost.

End-to-end, model reloaded before every question. The vector run was repeated on this
branch and reproduced the earlier baseline exactly (21 of 35, same failures), so the
difference is hybrid's:

| | Vector | Hybrid |
|---|---|---|
| Answers passing | **21 / 35** | 18 / 35 |
| Answerable questions refused | **9** | 10 |
| Citation precision | **0.815** | 0.800 |

Hybrid fixed three answers - `co-payment` (a retrieval miss before),
`specified-disease-waiting` and `icu-limit` (both refusals before). It broke six.
Four are refusals with an answer page ranked 1st or 2nd (`ped-waiting-period`,
`pre-hospitalisation`, `post-hospitalisation`, `maternity-excluded` - the last with
all five passages on answer pages). Two are answers missing the expected wording
(`modern-treatments`, `claim-documents-deadline`). Retrieval did not cause these; the
model's decision to refuse is sensitive to which passages it sees, and fusion changes
them. Refusals are the problem to fix first, and hybrid is re-measured after.

Tried and rejected:

| Variant | Result |
|---|---|
| `ts_rank_cd` (rewards words near each other) | 14 / 31 alone |
| Dropping words found in many chunks, as a stand-in for BM25's rare-word weighting | worse at every cut-off tried |
| Candidate depth 10 / 20 / 40, RRF constant 10 / 30 / 60 | 28 / 31 throughout, MRR within 0.02 |

## Consequences
- **The filename gain is partly a property of this corpus and question set.** The
  questions name products, and these filenames happen to contain product names. A
  corpus of files called `20240930T172140.pdf` - one of the fourteen is - gets nothing
  from it. The general version is document metadata (product, insurer, effective date)
  captured at ingest and searched deliberately; the filename is a cheap stand-in.
- `ts_rank` is not BM25. It gives a common word like "policy" the same weight as
  "cataract". At 1,570 chunks, the fusion absorbs this; on a larger corpus, a proper
  BM25 implementation is the next thing to measure.
- There is no keyword index. The match spans two tables, which no index can serve, and
  at this size the scan takes about 10 ms. At scale, the filename's words move into a
  chunks column and get a GIN index.
- Filenames written in CamelCase (`ArogyaSanjeevani-PolicyDocument.pdf`) are not split
  into words, so they contribute nothing.
- Hybrid runs both searches: median 29.8 ms against 16.7 ms for vector alone (105
  queries), against 11-17 seconds of generation.
- Shipping a component that is not the default is a cost: two code paths to test and
  keep honest. It is kept because the retrieval gain is real and measured, and the
  switch makes the end-to-end rerun a one-flag change.
- 31 questions is a small set. Moving from 27 to 30 is three questions; the rank
  regressions are real too, and three questions is also their size.
