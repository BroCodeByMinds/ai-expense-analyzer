from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddingService(ABC):

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass
    
