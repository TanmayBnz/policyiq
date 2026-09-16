# Measured numbers

Quote these in interviews. Re-measure when anything in the path changes.

## Embeddings — `bge-small-en-v1.5` via ONNX Runtime, CPU only

Measured 2026-09-14 on the development machine (no GPU used).

| Metric | Value |
|---|---|
| Batch throughput | ~12 chunks/sec on real ~950-character chunks |
| Full corpus ingest | 874 chunks in 71.8s, including parsing and database writes |
| Single query embedding | 3.3 ms |

**Correction, 2026-09-16.** This table first said 230 chunks/sec. That figure came from
embedding a 52-character sentence repeated 200 times; real chunks are about twenty
times longer, and transformer cost grows with input length. Re-measured on real chunk
text, and confirmed independently by the full ingest time.
| Vector dimension | 384 |
| Dependency footprint | ~100MB (ONNX) vs ~2.5GB (PyTorch) |

**The point:** embedding runs on CPU fast enough that no GPU is needed for query, and
ingestion is a one-off cost per document, which means the service deploys to ordinary cluster nodes. PyTorch was
rejected because it is a training stack — at inference time it costs 2.5GB of image
size and buys nothing.

## Chunking — clause-aware, target 1200 chars, overlap 150

Measured 2026-09-15 across the full corpus.

| Metric | Value |
|---|---|
| Corpus | 10 documents, 268 pages |
| Chunks produced | 874 |
| Chunks per page | 3.3 |
| Chunk size | min 95, median 998, max 1346 |
| Chunks over ceiling (1351) | 0 of 874 |

Before the review fixes: 16% of chunks exceeded the target, largest 2844 (2.4x).

## Corpus and chunking now — 2026-09-16

| Metric | Value |
|---|---|
| Corpus | 14 documents, 516 pages |
| Chunks | 1,570, each opening with its policy section |
| Chunks carrying a section label | 1,502 (95.7%) |
| Chunk body size | min 5 (a bare page number), median 1,046, max 1,346 |
| Chunks over ceiling (1351) | 0 |
| Full re-ingest | 2 min 36 s |

## Search and evaluation — 2026-09-16

| Metric | Value |
|---|---|
| Vector search, top 5, 1,570 chunks | median 18.2 ms, p95 21.2 ms (105 queries, embedding included) |
| Evaluation run, 35 questions, model reloaded per question | 660–674 s (median 16.5 s per question) |
| Retrieval hit@5 / MRR / precision@5 | 0.871 / 0.739 / 0.523 |
| Answers passing | 21 of 35 |
| Citation precision | 0.815 |
| Answers differing between two identical cold runs | 0 of 35 |
| Pass/fail flips between two identical runs with the model left loaded | 4 of 33 |

## Known data-quality gaps

| Metric | Value |
|---|---|
| Chunks starting with an unstripped page footer | 25, all in one document |
| Chunks with a body under 80 characters | 9 (mostly bare page numbers) |
