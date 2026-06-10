from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List

from app.models.responses.retrieval_result import RetrievalResult


class BaseVectorStore(ABC):

    async def store_documents(self, documents:List[Document], embeddings: List[List[float]]):
        pass

    async def search_similar_documents(self, query_embeddings:List[float], top_k:int) -> RetrievalResult:
        pass