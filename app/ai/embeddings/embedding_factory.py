from app.ai.embeddings.base_embeding_service import BaseEmbeddingService
from app.ai.embeddings.ollama_embedding_service import OllamaEmbeddingService
from app.ai.embeddings.openai_embedding_service import OpenAIEmbeddingService


class EmbeddingFactory:

    @staticmethod
    def get_embedding_service(provider: str) -> BaseEmbeddingService:
        """
        Resolve and return the appropriate embedding service
        implementation for the configured provider.
        """

        # Step 1: Map supported embedding providers to their
        # corresponding service implementations.
        embedding_mapping = {
            "openai": OpenAIEmbeddingService(),
            "ollama": OllamaEmbeddingService()
        }

        # Step 2: Retrieve the embedding service for the
        # requested provider.
        embedding_service : BaseEmbeddingService = embedding_mapping.get(provider.lower())

        # Step 3: Validate that the provider is supported.
        if not embedding_service:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        # Step 4: Return the resolved embedding service.
        return embedding_service
