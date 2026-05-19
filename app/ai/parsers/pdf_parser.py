from app.ai.parsers.base_parser import BaseParser
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from app.core.logging import setup_logger
from pypdf import PdfReader


logger = setup_logger(__name__)

class PDFParser(BaseParser):

    async def parse(self, file_path) -> List[Document]:
        logger.info(f"Starting PDF Parsing : {file_path}")
        
        documents : List[Document] = []
        pdf_path : Path = Path(file_path)

        if not pdf_path.exists():
            logger.error(f"FDF File not found in the given path : {file_path}")
        
        try:

            reader = PdfReader(file_path)

            logger.info(f"PDF loaded successfully. Total pages: {len(reader.pages)}")

            for page_number, page in enumerate(reader.pages):

                logger.info(f"Parsing page: {page_number + 1}")

                extracted_text = (page.extract_text())

                cleaned_text = self._clean_text(extracted_text)

                if not cleaned_text.strip():

                    logger.warning(f"Skipping empty page: {page_number + 1}")
                    continue

                document = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number + 1,
                        "file_path": str(pdf_path),
                        "file_type": "pdf",
                    }
                )

                documents.append(document)

            logger.info(f"PDF parsing completed. Total documents created: {len(documents)}")

            return documents

        except Exception as exception:
            logger.exception(f"Failed to parse PDF: {file_path}")
            raise exception
        

    @staticmethod
    def _clean_text(text: str) -> str:

        if not text:
            return ""

        return (text
            .replace("\n", " ")
            .replace("\t", " ")
            .strip()
        )