# PolicyIQ

Question answering over insurance policy documents, with citations that point at a
real page you can open and check.

Runs entirely on local infrastructure. No third-party API is called and no document
text leaves the machine — which is not a limitation worked around but the deployment
model a regulated insurer would require.

> **Status: in progress.** Ingestion, vector and keyword search, grounded answers with citations,
> and an evaluation harness are built and tested. Deployment is next. See [Roadmap](#roadmap) for what exists today versus what is planned, and
> [Evaluation](#evaluation) for how well it currently works.

---

## Why this exists

A general "upload a PDF and ask questions" demo cannot be evaluated, because there is
no correct answer to compare against. This is deliberately narrow instead: a single
standardised insurance product (Arogya Sanjeevani health policies, published by
multiple Indian insurers under a common IRDAI specification).

That narrowness buys three things. Policy documents have genuine clause structure, so
chunking decisions have a defensible basis. Questions about them have unambiguous
answers, so retrieval quality can actually be measured. And the same question is
answerable across several insurers' documents, which makes a meaningful evaluation set
cheap to build.

The corpus is 14 documents: nine Arogya Sanjeevani wordings and prospectuses, plus
five other health policy wordings (Niva Bupa ReAssure 2.0 and 3.0, HDFC ERGO Optima
Secure, Star Health Assure, ICICI Lombard Elevate) whose terms genuinely differ — which
is what makes some questions discriminating.

## Architecture

```
                       ┌──────────────────────────┐
   PDF policies ──────▶│  POST /v1/ingest         │
                       │  parse → clause-aware    │
                       │  chunk → embed → store   │
                       └───────────┬──────────────┘
                                   ▼
                       ┌──────────────────────────┐
                       │  PostgreSQL + pgvector   │
                       │  vectors + full-text     │
                       └───────────┬──────────────┘
                                   ▲
   question ───────────▶┌──────────┴──────────────┐
                        │  POST /v1/query          │
                        │  vector search (or fused │
                        │  with keyword), then     │
                        │  generated from excerpts │
                        │  → answer + citations    │
                        └──────────────────────────┘
```

| Component | Choice | Why |
|---|---|---|
| Embeddings | `bge-small-en-v1.5` via ONNX Runtime | About 12 chunks/sec on CPU for real ~1,000-character chunks, and 3.5ms per query — no GPU needed. PyTorch would add ~2.5GB to the image and buy nothing at inference time. |
| Generation | Ollama on the host, 3B instruct model | Zero cost, zero egress. Sized to fit 4GB of VRAM rather than spilling to CPU mid-demo. |
| Model layer | `LLMProvider` interface | Clients arrive with incompatible constraints — some locked into a managed cloud model, some unable to use a hosted model at all. Swapping providers must not touch retrieval. |
| Storage | PostgreSQL + pgvector | One datastore for both vector and keyword search, rather than a separate vector database and search engine. |
| Retrieval | Vector search by default; vector and keyword search merged by reciprocal rank fusion behind `RETRIEVAL_MODE=hybrid` | They miss different questions: embeddings barely register a product name or a date, keywords miss paraphrase. Fused, 30 of 31 answerable questions have an answer page in the top 5, against 27 — but end-to-end answers got worse (18 of 35 against 21), so hybrid is not the default yet. |
| Data access | psycopg with raw SQL | The vector query is the interesting part of this system; it should be readable directly rather than through a query builder. |

## Design decisions worth reading

Recorded as ADRs in [`docs/adr/`](docs/adr/):

- **[Clause-aware chunking](docs/adr/0001-chunking-strategy.md)** — why fixed-width
  chunking is actively dangerous in a policy document, and why overlap deliberately
  stops at page boundaries to keep citations exact.
- **[Evaluation without a model judge](docs/adr/0002-evaluation-harness.md)** — a
  golden set keyed on pages rather than chunks, retrieval and answers scored
  separately, and why every question runs against a freshly loaded model.
- **[Hybrid search, fused by rank](docs/adr/0003-hybrid-search.md)** — why keyword
  search matches any word rather than all, why rankings are fused by position rather
  than score, and why better retrieval did not yet mean better answers.

Measured figures are in [`docs/measurements.md`](docs/measurements.md).

## Running it

Requires Docker, and [Ollama](https://ollama.com) on the host bound to `0.0.0.0`.

```bash
cp .env.example .env
ollama pull qwen2.5:3b-instruct
make up
curl localhost:8000/readyz
```

Tests:

```bash
uv venv && uv pip install -e ".[dev]"
make test
```

The test suite generates its own specimen policy document, so it runs without the
real corpus — which is deliberate, since those documents are not redistributable and
CI will never have them.

## Evaluation

35 questions in [`evaluation/questions.json`](evaluation/questions.json), each
recording the document pages its answer is on. Four have no answer in the corpus;
the correct response to those is a refusal.

```bash
python -m policyiq.evaluation run --retrieval-only   # seconds
python -m policyiq.evaluation run --retrieval-only --retrieval hybrid   # compare
python -m policyiq.evaluation run                    # ~11 minutes, repeatable
```

Baseline, 2026-09-16 — `qwen2.5:3b-instruct`, top 5, chunks of 1,200 characters with
section labels, model reloaded before every question:

| | |
|---|---|
| Retrieval: an answer page in the top 5 | 27 of 31 (0.871) |
| Retrieval: mean reciprocal rank | 0.739 |
| Answers passing | 21 of 35 (0.60) |
| Out-of-scope questions correctly refused | 4 of 4 |
| Citations pointing at a page that holds the answer | 81.5% |

The largest failure is not wrong answers but refusals: 9 answerable questions got
"the provided policy documents do not cover this" — 6 of them with an answer page
among the five passages the model was given, and two of those with the answer on all
five. Three of the four retrieval misses are questions that name a product or
a policy date, which vector search does not use.

Hybrid search, 2026-09-17, same settings and the vector run repeated on the same code
(21 of 35 again, with the same failures):

| | Vector | Hybrid |
|---|---|---|
| Retrieval: an answer page in the top 5 | 27 of 31 | **30 of 31** |
| Retrieval: mean reciprocal rank | 0.739 | **0.769** |
| Answers passing | **21 of 35** | 18 of 35 |
| Answerable questions refused | **9** | 10 |
| Citation precision | **81.5%** | 80.0% |

Better retrieval, worse answers. Hybrid fixed three answers (co-payment,
specified-disease waiting, ICU limit) and broke six, four of them by refusing with an
answer page already ranked first or second. The fused passages are different, not
worse, and the 3B model's refusals are fragile enough that different passages flip
them. Vector search stays the default until the refusal problem is fixed; the
comparison is then rerun. Details in [ADR 0003](docs/adr/0003-hybrid-search.md).

Two identical runs produce identical answers to all 35 questions. That took reloading
the model before every question: left loaded, two runs of the same code at temperature
0 disagreed on 4 pass/fail results, which is noise larger than most changes worth
measuring. Details in [ADR 0002](docs/adr/0002-evaluation-harness.md).

## Roadmap

| | Component | State |
|---|---|---|
| ✅ | Page-accurate PDF extraction, intake validation | 516 pages across 14 documents |
| ✅ | Clause-aware chunking with section labels | 1,570 chunks, median body 1,046 chars |
| ✅ | Local ONNX embeddings | ~12 chunks/sec on CPU |
| ✅ | PostgreSQL + pgvector schema | |
| ✅ | Liveness / readiness endpoints | |
| ✅ | Ingestion pipeline and endpoint | |
| ✅ | Vector search | |
| ✅ | Keyword search and reciprocal rank fusion (opt-in) | 30 of 31 in the top 5, from 27; answers 18 of 35, from 21 |
| ✅ | Grounded answers with citations resolved from the database | |
| ✅ | Evaluation harness over a golden question set | see [Evaluation](#evaluation) |
| ⬜ | CI/CD, Helm chart, Kubernetes deployment | |

## Notes on the corpus

Source documents are not committed — they are publicly available insurer
publications, but not mine to redistribute. `data/policies/` is gitignored.

One document was rejected during intake: its PDF contained the full policy text
duplicated into every page's content stream, roughly ten times the density of every
other file, with several pages byte-identical. Ingesting it would have flooded
retrieval with near-duplicates and made page citations untraceable. Validating
documents before loading them, rather than debugging retrieval quality afterwards, is
now part of the pipeline.
