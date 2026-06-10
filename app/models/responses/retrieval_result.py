from typing import Any, List

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    content: str
    metadata: dict[str, Any]
    similarity_score: float


class RetrievalResult(BaseModel):
    chunks: List[RetrievedChunk]