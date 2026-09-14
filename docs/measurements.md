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

## To be filled in

- [ ] Chunks per document, median chunk size (Task 4)
- [ ] Vector search latency (Task 7)
- [ ] End-to-end query latency (Task 9)
- [ ] Generation latency by model (Task 8)
