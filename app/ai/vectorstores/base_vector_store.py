from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List


class BaseVectorStore(ABC):

    async def store_documents(self, documents:List[Document], embeddings: List[List[float]]):
        pass
    