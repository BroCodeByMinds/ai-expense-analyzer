from typing import List
from langchain_core.documents import Document
import chromadb
from app.ai.vectorstores.base_vector_store import BaseVectorStore
from app.core.logging import setup_logger

logger = setup_logger(__name__)


class ChromaService(BaseVectorStore):
    def __init__(self):
        self.client = chromadb.PersistentClient(path=".chroma_db")

        self.collection = self.client.get_or_create_collection(name="expense_documents")

    
    async def store_documents(self, documents, embeddings):
        logger.info(f"{self.__class__.__name__} | Starting vector storage. Total documents: {len(documents)}")

        ids = [f"doc_{index}" for index in range(len(documents))]

        texts : List[str] = [document.page_content for document in documents]

        metadatas = [document.metadata for document in documents]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"{self.__class__.__name__} | Documents stored successfully in ChromaDB.")