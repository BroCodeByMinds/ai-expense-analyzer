from app.ai.embeddings.base_embeding_service import BaseEmbeddingService
from app.ai.embeddings.ollama_embedding_service import OllamaEmbeddingService
from app.ai.embeddings.openai_embedding_service import OpenAIEmbeddingService


class EmbeddingFactory:

    @staticmethod
    def get_embedding_service(provider: str) -> BaseEmbeddingService:
        embedding_mapping = {
            "openai": OpenAIEmbeddingService(),
            "ollama": OllamaEmbeddingService()
        }
        embedding_service = embedding_mapping.get(provider.lower())

        if not embedding_service:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        return embedding_service
