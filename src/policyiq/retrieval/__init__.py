"""Retrieval.

Vector search (vector.py) and keyword search (keyword.py) each rank chunks their own
way; fusion.py merges the two rankings; search.py chooses between vector-only and the
fused result and is the only entry point the rest of the code calls. The boundary exists
so that replacing what is inside this package does not require touching ingestion,
answering, or the routes.
"""
