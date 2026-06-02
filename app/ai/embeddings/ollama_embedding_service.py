from typing import List

from app.ai.embeddings.base_embeding_service import BaseEmbeddingService
from langchain_community.embeddings import OllamaEmbeddings
from app.core.logging import setup_logger

logger = setup_logger(__name__)



class OllamaEmbeddingService(BaseEmbeddingService):
    def __init__(self):
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    async def generate_embeddings(self, texts: List[str]):
        logger.info(f"{self.__class__.__name__} | Generating embeddings. Total texts: {len(texts)}")

        embeddings = self.embedding_model.embed_documents(texts=texts)

        logger.info(f"{self.__class__.__name__} | Embedding generation completed.")

        return embeddings