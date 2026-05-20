from app.ai.vectorstores.base_vector_store import BaseVectorStore
from app.ai.vectorstores.chroma_service import ChromaService
from app.core.logging import setup_logger

logger = setup_logger(__name__)

class VectorStoreFactory:
    @staticmethod
    def get_vector_store(provider: str) -> BaseVectorStore:

        vector_store_mapping = {
            "chroma_db": ChromaService()
        }

        store_service =  vector_store_mapping.get(provider)

        if not store_service:
            logger.error(f"Vector Store is not implemented for provoider{provider}")
        
        return store_service