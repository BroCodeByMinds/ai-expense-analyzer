from app.ai.parsers.base_parser import BaseParser
from app.ai.parsers.pdf_parser import PDFParser
from app.core.logging import setup_logger

logger = setup_logger(__name__)

class ParserFactory:

    @staticmethod
    def get_parser(file_type: str) -> BaseParser:
        """
        Resolve and return the appropriate parser implementation
        for the provided file type.
        """

        # Step 1: Map supported file types to their
        # corresponding parser implementations.
        parser_mapping = {
            "pdf": PDFParser(),
            # "csv": CSVParser()
        }

        # Step 2: Retrieve the parser implementation
        # for the requested file type.
        parser : BaseParser = parser_mapping.get(file_type.lower())

        # Step 3: Log an error if no parser has been
        # configured for the given file type.
        if not parser:
            logger.error(f"Parser is not yet configured for the file type : {file_type.upper()}")

        # Step 4: Return the resolved parser instance.
        return parser