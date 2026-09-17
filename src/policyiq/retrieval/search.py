from policyiq.config import settings
from policyiq.retrieval.fusion import reciprocal_rank_fusion
from policyiq.retrieval.keyword import keyword_search
from policyiq.retrieval.vector import RetrievedChunk, vector_search

MODES = ("vector", "hybrid")


def retrieve(question: str, top_k: int, mode: str | None = None) -> list[RetrievedChunk]:
    """The one entry point answering and evaluation use.

    `mode` defaults to settings.retrieval_mode. Vector-only is kept rather than deleted
    so that the gain from hybrid stays measurable: the evaluation report records the
    mode, and the same golden set can be run both ways.
    """
    mode = mode or settings.retrieval_mode
    if mode == "vector":
        return vector_search(question, top_k)
    if mode == "hybrid":
        # DECISION: fuse from more candidates than will be returned. A chunk that is
        # 7th for vector search and 2nd for keyword search can deserve a top-5 place,
        # and it is only seen if each retriever is asked for more than 5.
        #
        # 20 is the usual order of magnitude, not a tuned value: 10, 20 and 40 all gave
        # 28/31 on the golden set before the filename was indexed. Deeper is not free
        # for a system with more documents - a long keyword list is mostly weak matches,
        # and at depth 40 with k=10 MRR fell to 0.696 from 0.712.
        #
        # max(...) because asking for 30 results from 20 candidates each could return
        # fewer than asked. It also means top_k above 20 draws on a deeper pool, so a
        # top_k of 25 is not guaranteed to extend a top_k of 5 unchanged. At or below
        # 20 it is, which covers every real request.
        depth = max(settings.retrieval_candidates, top_k)
        return reciprocal_rank_fusion(
            [vector_search(question, depth), keyword_search(question, depth)],
            top_k,
            settings.rrf_k,
        )
    # Loud rather than falling back. A bad RETRIEVAL_MODE is already refused when the
    # settings load; this catches a bad `mode` argument, which silently meaning "vector"
    # would make a comparison report a mode it did not run.
    raise ValueError(f"unknown retrieval mode {mode!r}, expected one of {MODES}")
