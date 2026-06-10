from app.ai.vectorstores.base_vector_store import BaseVectorStore
from app.ai.vectorstores.chroma_service import ChromaService
from app.core.logging import setup_logger

logger = setup_logger(__name__)

class VectorStoreFactory:
    @staticmethod
    def get_vector_store(provider: str) -> BaseVectorStore:
        """
        Resolve and return the appropriate vector store
        implementation for the configured provider.
        """

        # Step 1: Map supported vector store providers to
        # their corresponding implementations.
        vector_store_mapping = {
            "chroma_db": ChromaService()
        }

        # Step 2: Retrieve the vector store implementation
        # for the requested provider.
        store_service : BaseVectorStore =  vector_store_mapping.get(provider)

        # Step 3: Validate that the provider is supported.
        if not store_service:
            logger.error(f"Vector Store is not implemented for provoider{provider}")
        
        # Step 4: Return the resolved vector store instance.
        return store_service