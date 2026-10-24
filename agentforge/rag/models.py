from pydantic import BaseModel, Field
from typing import Any

class Document(BaseModel):
    source: str
    text: str
    page: int | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

class Chunk(BaseModel):
    id:str
    document_id: str
    chunk_index: int
    text: str
    source: str
    page: int | None = None


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    method: str = "hybrid"