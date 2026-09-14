"""Request and response bodies for the HTTP API.

These live apart from the pipeline so that the wire format can change without
touching ingestion logic, and so FastAPI can generate the OpenAPI schema from one
place.
"""

from pydantic import BaseModel


class IngestResult(BaseModel):
    document_id: int
    filename: str
    page_count: int
    chunk_count: int
