from typing import List
from langchain_core.documents import Document

from app.ai.chunking.chunk_service import ChunkService
from app.ai.parsers.base_parser import BaseParser
from app.ai.parsers.parser_factory import ParserFactory

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

        return chunked_docs