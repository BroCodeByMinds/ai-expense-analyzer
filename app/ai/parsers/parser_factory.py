from app.ai.parsers.base_parser import BaseParser
from app.ai.parsers.pdf_parser import PDFParser
from app.core.logging import setup_logger

logger = setup_logger(__name__)

class ParserFactory:

    @staticmethod
    def get_parser(file_type: str) -> BaseParser:
        parser_mapping = {
            "pdf": PDFParser(),
            # "csv": CSVParser()
        }

        parser = parser_mapping.get(file_type.lower())

        if not parser:
            logger.error(f"Parser is not yet configured for the file type : {file_type.upper()}")

        return parser