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

        file_type : str = get_file_extension(file_path)

        logger.info(
            f"{self.__class__.__name__} | "
            f"Starting document parsing. "
            f"File path: {file_path}, "
            f"File type: {file_type}"
        )

        parser : BaseParser = ParserFactory.get_parser(file_type=file_type)

        logger.info(
            f"{self.__class__.__name__} | "
            f"Parser resolved successfully. "
            f"Parser: {parser.__class__.__name__}"
        )

        documents : List[Document] = await parser.parse(file_path=file_path)

        logger.info(
            f"{self.__class__.__name__} | "
            f"Document parsing completed successfully. "
            f"Total documents parsed: "
            f"{len(documents)}"
        )

        chunk_service : ChunkService = ChunkService()

        chunked_docs : List[Document] = await chunk_service.chunk_documents(documents=documents)

        logger.info(
            f"{self.__class__.__name__} | "
            f"Chunking completed. "
            f"Total chunks: "
            f"{len(chunked_docs)}"
        )

        embedding_service: BaseEmbeddingService = (EmbeddingFactory.get_embedding_service(provider="openai"))

        texts: List[str] = [chunk.page_content for chunk in chunked_docs]

        embeddings: List[List[float]] = (await embedding_service.generate_embeddings(texts=texts))

        logger.info(
            f"{self.__class__.__name__} | "
            f"Embedding generation completed successfully. "
            f"Total embeddings: "
            f"{len(embeddings)}"
        )

        logger.info(
            f"{self.__class__.__name__} | "
            f"Embedding dimension: "
            f"{len(embeddings[0])}"
        )

        store_service : BaseVectorStore = VectorStoreFactory.get_vector_store("chroma_db")

        store_service.store_documents(documents=chunked_docs, embeddings=embeddings)

        logger.info(
            f"{self.__class__.__name__} | "
            f"Documents stored successfully "
            f"in vector database."
        )

        return chunked_docs