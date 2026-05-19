from app.ai.parsers.parser_factory import ParserFactory
from app.core.logging import setup_logger
from app.utils.file_utils import get_file_extension


logger = setup_logger(__name__)


class IngestionService:

    async def parse_file(self, file_path: str):

        file_type : str = get_file_extension(file_path)

        logger.info(
            f"{self.__class__.__name__} | "
            f"Starting document parsing. "
            f"File path: {file_path}, "
            f"File type: {file_type}"
        )

        parser = ParserFactory.get_parser(file_type=file_type)

        logger.info(
            f"{self.__class__.__name__} | "
            f"Parser resolved successfully. "
            f"Parser: {parser.__class__.__name__}"
        )

        documents = await parser.parse(file_path=file_path)

        logger.info(
            f"{self.__class__.__name__} | "
            f"Document parsing completed successfully. "
            f"Total documents parsed: "
            f"{len(documents)}"
        )

        return documents