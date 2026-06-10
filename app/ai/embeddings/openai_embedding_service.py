from app.ai.embeddings.base_embeding_service import BaseEmbeddingService
from app.core.logging import setup_logger
from langchain_openai import OpenAIEmbeddings
from typing import List

logger = setup_logger(__name__)

class OpenAIEmbeddingService(BaseEmbeddingService):
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for the provided text inputs
        using the configured embedding model.
        """
        logger.info(
            f"{self.__class__.__name__} | "
            f"Starting embedding generation. "
            f"Total texts: {len(texts)}"
        )

        # Step 1: Generate embeddings for the provided text
        # inputs using the configured embedding model.
        embeddings : list[list[float]] = await self.embedding_model.aembed_documents(texts=texts)
        logger.info(
            f"{self.__class__.__name__} | "
            f"Embedding generation completed."
        )

        # Step 2: Return the generated embeddings for
        # downstream vector storage and retrieval.
        return embeddings
    