# Measured numbers

Quote these in interviews. Re-measure when anything in the path changes.

## Embeddings — `bge-small-en-v1.5` via ONNX Runtime, CPU only

Measured 2026-09-14 on the development machine (no GPU used).

| Metric | Value |
|---|---|
| Batch throughput | 230 chunks/sec (200 chunks in 0.87s) |
| Single query embedding | 3.3 ms |
| Vector dimension | 384 |
| Dependency footprint | ~100MB (ONNX) vs ~2.5GB (PyTorch) |

**The point:** embedding runs on CPU fast enough that no GPU is needed for ingestion
or query, which means the service deploys to ordinary cluster nodes. PyTorch was
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

## To be filled in
- [ ] Vector search latency (Task 7)
- [ ] End-to-end query latency (Task 9)
- [ ] Generation latency by model (Task 8)
