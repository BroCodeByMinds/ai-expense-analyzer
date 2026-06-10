from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from app.core.logging import setup_logger

logger = setup_logger(__name__)

class ChunkService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap : int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                            chunk_overlap=self.chunk_overlap)
        
    
    async def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks suitable for embedding
        generation and semantic search.
        """
        logger.info(f"{self.__class__.__name__} | Starting document chunking. Total documents: {len(documents)}")

        # Step 1: Split the input documents into smaller chunks
        # using the configured text splitter.
        chunked_documents : List[Document] = self.text_splitter.split_documents(documents=documents)

        # Step 2: Assign a unique chunk index to each chunk
        # for traceability during retrieval and debugging.
        for index, chunk in enumerate(chunked_documents):
            chunk.metadata["chunk_index"] = index 

        # Step 3: Return the chunked documents for downstream
        # embedding generation and vector storage.
        return chunked_documents
            

