"""Request and response bodies for the HTTP API.

These live apart from the pipeline so that the wire format can change without
touching ingestion logic, and so FastAPI can generate the OpenAPI schema from one
place.
"""

from pydantic import BaseModel, Field, field_validator


class IngestResult(BaseModel):
    document_id: int
    filename: str
    page_count: int
    chunk_count: int


class Citation(BaseModel):
    """Where an answer came from.

    Every field is read from the database row of a chunk that was actually retrieved.
    None of it is parsed out of the model's reply - a model asked to write its own
    citations will produce page numbers that do not exist, and they look identical to
    real ones.
    """

    document_id: int
    filename: str
    page_number: int
    chunk_index: int
    excerpt: str


class QueryRequest(BaseModel):
    question: str
    # Bounded below because top_k reaches a SQL LIMIT, and above because every extra
    # chunk costs context window and dilutes the model's attention.
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """A whitespace-only question is not a question.

        Left unchecked it would be embedded, retrieve essentially arbitrary chunks, and
        produce a confident answer to nothing. Rejecting at the boundary turns that into
        a 422 naming the field.
        """
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
