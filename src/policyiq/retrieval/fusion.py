from collections.abc import Sequence
from dataclasses import replace

from policyiq.retrieval.vector import RetrievedChunk

# The constant from the paper that introduced the method (Cormack, Clarke and Buettcher,
# 2009), and the usual default. It sets how much more the first place is worth than the
# tenth: with k=60 that is 1/61 against 1/70, a gentle slope, so agreement between
# retrievers counts for more than either one's top pick.
#
# Not tuned, deliberately. On the golden set k=10, 30 and 60 at 20 candidates gave the
# same 28/31 before the filename was indexed, with MRR moving by 0.001. Tuning it on
# 31 questions would fit those questions, not the method.
DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], top_k: int, k: int = DEFAULT_K
) -> list[RetrievedChunk]:
    """Merge several best-first rankings into one.

    Each chunk scores the sum of 1 / (k + rank) over every ranking it appears in, where
    rank starts at 1. A chunk absent from a ranking gets nothing from it.

    DECISION: combine positions, not scores. Vector similarity runs roughly 0.5 to 0.9
    here; ts_rank runs roughly 0.01 to 0.1 and has no fixed ceiling. Adding them, or
    even averaging them after rescaling, lets whichever scale is wider decide the result,
    and the scales move with every change of question. Positions have the same meaning
    in both lists. The cost is that a clear winner - a chunk scoring far above the rest -
    counts no more than a narrow one.

    The returned chunks carry their fused score in `score`, replacing the retriever's
    own. It is only meaningful for ordering within this result.
    """
    if k <= 0:
        # k=0 makes first place worth 1/1 against second's 1/2: one retriever's top pick
        # would outvote agreement. Negative k divides by zero at rank -k.
        raise ValueError("k must be positive")

    fused: dict[int, float] = {}
    chunks: dict[int, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            # Keyed by chunk id, so the same chunk found by both retrievers is one
            # result with two contributions - not two results. Keying by object
            # would fail: each retriever builds its own RetrievedChunk, and their
            # scores differ, so the two are never equal.
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(chunk.chunk_id, chunk)

    # Ties broken by chunk id, as in both retrievers, so the order is total and a
    # rerun gives the same list. Ties are ordinary: a chunk ranked 3rd by one retriever
    # and absent from the other scores exactly what another does in the mirror position.
    order = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))
    return [replace(chunks[i], score=fused[i]) for i in order[:top_k]]
