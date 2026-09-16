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


settings = Settings()
