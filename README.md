# PolicyIQ

Question answering over insurance policy documents, with citations that point at a
real page you can open and check.

Runs entirely on local infrastructure. No third-party API is called and no document
text leaves the machine — which is not a limitation worked around but the deployment
model a regulated insurer would require.

> **Status: in progress.** Ingestion and the retrieval foundations are built and
> tested. Hybrid search, the evaluation harness and the query endpoint are next. See
> [Roadmap](#roadmap) for what exists today versus what is planned.

---

## Why this exists

A general "upload a PDF and ask questions" demo cannot be evaluated, because there is
no correct answer to compare against. This is deliberately narrow instead: a single
standardised insurance product (Arogya Sanjeevani health policies, published by
multiple Indian insurers under a common IRDAI specification).

That narrowness buys three things. Policy documents have genuine clause structure, so
chunking decisions have a defensible basis. Questions about them have unambiguous
answers, so retrieval quality can actually be measured. And the same question is
answerable across ten different insurers' documents, which makes a meaningful
evaluation set cheap to build.

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
                       │  vectors + keyword index │
                       └───────────┬──────────────┘
                                   ▲
   question ───────────▶┌──────────┴──────────────┐
                        │  POST /v1/query          │
                        │  vector + keyword search │
                        │  fused, then generated   │
                        │  → answer + citations    │
                        └──────────────────────────┘
```

| Component | Choice | Why |
|---|---|---|
| Embeddings | `bge-small-en-v1.5` via ONNX Runtime | 230 chunks/sec on CPU, no GPU needed. PyTorch would add ~2.5GB to the image and buy nothing at inference time. |
| Generation | Ollama on the host, 3B instruct model | Zero cost, zero egress. Sized to fit 4GB of VRAM rather than spilling to CPU mid-demo. |
| Model layer | `LLMProvider` interface | Clients arrive with incompatible constraints — some locked into a managed cloud model, some unable to use a hosted model at all. Swapping providers must not touch retrieval. |
| Storage | PostgreSQL + pgvector | One datastore for vector and keyword search rather than running a separate vector database. |
| Data access | psycopg with raw SQL | The vector query is the interesting part of this system; it should be readable directly rather than through a query builder. |

## Design decisions worth reading

Recorded as ADRs in [`docs/adr/`](docs/adr/):

- **[Clause-aware chunking](docs/adr/0001-chunking-strategy.md)** — why fixed-width
  chunking is actively dangerous in a policy document, and why overlap deliberately
  stops at page boundaries to keep citations exact.

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

## Roadmap

| | Component | State |
|---|---|---|
| ✅ | Page-accurate PDF extraction | 268 pages across 10 documents |
| ✅ | Clause-aware chunking | 874 chunks, median 998 chars |
| ✅ | Local ONNX embeddings | 230 chunks/sec on CPU |
| ✅ | PostgreSQL + pgvector schema | |
| ✅ | Liveness / readiness endpoints | |
| ⬜ | Ingestion pipeline and endpoint | |
| ⬜ | Hybrid search with reciprocal rank fusion | |
| ⬜ | Grounded answers with verified citations | |
| ⬜ | Evaluation harness over a golden question set | |
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
