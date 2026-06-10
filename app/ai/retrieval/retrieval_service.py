from typing import List

from app.ai.embeddings.base_embeding_service import BaseEmbeddingService
from app.ai.embeddings.embedding_factory import EmbeddingFactory
from app.ai.vectorstores.base_vector_store import BaseVectorStore
from app.ai.vectorstores.vector_store_factory import VectorStoreFactory
from app.models.responses.retrieval_result import RetrievalResult


class RetrievalService:

    async def search(self, query: str, top_k:int= 5) -> RetrievalResult:
        # Step 1: Resolve the configured embedding provider.
        embedding_service : BaseEmbeddingService = EmbeddingFactory.get_embedding_service(provider='ollama')

        # Step 2: Generate embedding for the user query.
        query_embedding: List[List[float]] = await embedding_service.generate_embeddings(texts=[query])

        # Step 3: Resolve the configured vector store.
        vector_store: BaseVectorStore = VectorStoreFactory.get_vector_store('chroma_db')

        # Step 4: Perform semantic search using the query
        # embedding and retrieve the most relevant chunks.
        retrieval_result: RetrievalResult = await vector_store.search_similar_documents(
            query_embeddings=query_embedding, top_k=top_k)

        # Step 5: Return the retrieved chunks.
        return retrieval_result

