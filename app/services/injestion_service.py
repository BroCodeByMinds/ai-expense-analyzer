from typing import List
from langchain_core.documents import Document

from app.ai.chunking.chunk_service import ChunkService
from app.ai.embeddings.base_embeding_service import BaseEmbeddingService
from app.ai.embeddings.embedding_factory import EmbeddingFactory
from app.ai.parsers.base_parser import BaseParser
from app.ai.parsers.parser_factory import ParserFactory

from app.ai.vectorstores.base_vector_store import BaseVectorStore
from app.ai.vectorstores.vector_store_factory import VectorStoreFactory
from app.utils.file_utils import get_file_extension
from app.core.logging import setup_logger

logger = setup_logger(__name__)


class IngestionService:

    async def parse_file(self, file_path: str) -> List[Document]:
        # Step 1: Identify the file type from the file extension
        # Example: pdf, csv, xlsx
        file_type : str = get_file_extension(file_path)

        # Step 2: Resolve the appropriate parser implementation
        # using the factory pattern
        parser : BaseParser = ParserFactory.get_parser(file_type=file_type)

        # Step 3: Parse the source file and convert it into
        # LangChain Document objects
        documents : List[Document] = await parser.parse(file_path=file_path)

        # Step 4: Split documents into smaller chunks suitable
        # for embedding generation and semantic search
        chunk_service : ChunkService = ChunkService()
        chunked_docs : List[Document] = await chunk_service.chunk_documents(documents=documents)

        # Step 5: Generate embeddings for each chunk using
        # the configured embedding provider
        embedding_service: BaseEmbeddingService = EmbeddingFactory.get_embedding_service(provider="ollama")
        texts: List[str] = [chunk.page_content for chunk in chunked_docs]
        embeddings: List[List[float]] = await embedding_service.generate_embeddings(texts=texts)

        # Step 6: Persist chunked documents, metadata, and
        # embeddings into the configured vector database
        store_service : BaseVectorStore = VectorStoreFactory.get_vector_store("chroma_db")
        await store_service.store_documents(documents=chunked_docs, embeddings=embeddings)

        return chunked_docs