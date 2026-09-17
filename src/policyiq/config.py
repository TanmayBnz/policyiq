from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://policyiq:policyiq@localhost:5432/policyiq"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b-instruct"
    # A 3B model on a 4GB GPU answers in seconds, but a cold load or a model that
    # has spilled to CPU can take far longer. Without a ceiling a wedged server
    # holds the request open indefinitely.
    ollama_timeout_seconds: float = 120.0
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    chunk_target_chars: int = 1200
    chunk_overlap_chars: int = 150
    retrieval_top_k: int = 5
    # "hybrid" fuses vector and keyword search; "vector" is vector search alone.
    # DECISION: vector stays the default. Hybrid finds an answer page for more
    # questions (30 of 31 against 27), but end-to-end answers got worse (18 of 35
    # against 21), because the 3B model refused more often with the fused passages.
    # The answer is what the user sees, so it decides. Hybrid is one env var away
    # (RETRIEVAL_MODE=hybrid) and becomes the default once refusals are fixed and
    # the comparison is rerun - see docs/adr/0003-hybrid-search.md.
    # A value outside the two is rejected at startup.
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    # How many results each retriever contributes to the fusion, and the RRF constant.
    # Both are explained where they are used, in policyiq/retrieval/.
    retrieval_candidates: int = 20
    rrf_k: int = 60


settings = Settings()
