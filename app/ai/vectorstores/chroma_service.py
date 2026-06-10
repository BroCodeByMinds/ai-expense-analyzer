from typing import Any, Dict, List
from langchain_core.documents import Document
import chromadb
from app.ai.vectorstores.base_vector_store import BaseVectorStore
from app.core.logging import setup_logger
from app.models.responses.retrieval_result import RetrievalResult, RetrievedChunk

logger = setup_logger(__name__)


class ChromaService(BaseVectorStore):
    def __init__(self):
        self.client = chromadb.PersistentClient(path=".chroma_db")

        self.collection = self.client.get_or_create_collection(name="expense_documents")

    
    async def store_documents(self, documents, embeddings):
        """
        Store document chunks, embeddings, and metadata in the
        configured ChromaDB collection.
        """
        logger.info(f"{self.__class__.__name__} | Starting vector storage. Total documents: {len(documents)}")

        # Step 1: Generate unique identifiers for each
        # document chunk to be stored in the vector database.
        ids = [f"doc_{index}" for index in range(len(documents))]

        # Step 2: Extract the chunk content from the document
        # objects for vector store persistence.
        texts : List[str] = [document.page_content for document in documents]

        # Step 3: Extract document metadata to support
        # filtering, traceability, and retrieval.
        metadatas = [document.metadata for document in documents]

        # Step 4: Store document chunks, embeddings, and
        # metadata in the configured ChromaDB collection.
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"{self.__class__.__name__} | Documents stored successfully in ChromaDB.")

    
    async def search_similar_documents(self, query_embeddings: List[float], top_k: int=5) -> RetrievalResult:
        """
        Retrieve the most relevant document chunks from the vector store
        based on semantic similarity.
        """
        # Perform similarity search against the stored embeddings
        # and return the top matching document chunks.
        results : Dict[str, Any] = self.collection.query(query_embeddings=query_embeddings, n_results=top_k)

        # Step 2: Transform the vector store response into the
        # application-specific retrieval response model.
        chunks : List[RetrievedChunk] = []
        for index, document in enumerate(results["documents"][0]):

            chunks.append(
                RetrievedChunk(
                    content=document,
                    metadata=results["metadatas"][0][index],
                    similarity_score=(
                        results["distances"][0][index]
                    )
                )
            )

        # Step 3: Return the retrieved document chunks along with
        # their metadata and similarity scores.
        return RetrievalResult(chunks=chunks)