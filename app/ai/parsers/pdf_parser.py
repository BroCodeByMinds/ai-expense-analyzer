from app.ai.parsers.base_parser import BaseParser
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from app.core.logging import setup_logger
from pypdf import PdfReader


logger = setup_logger(__name__)

class PDFParser(BaseParser):

    async def parse(self, file_path: str) -> List[Document]:
        """
        Parse a PDF file and convert each non-empty page into
        a LangChain Document object.
        """

        logger.info(f"Starting PDF Parsing : {file_path}")
        
        documents : List[Document] = []
        pdf_path : Path = Path(file_path)

        # Step 1: Validate that the PDF file exists before
        # attempting to load and parse it.
        if not pdf_path.exists():
            logger.error(f"FDF File not found in the given path : {file_path}")
        
        try:
            # Step 2: Load the PDF file and initialize the
            # PDF reader for page-level processing.
            reader: PdfReader = PdfReader(file_path)

            logger.info(f"PDF loaded successfully. Total pages: {len(reader.pages)}")

            # Step 3: Iterate through each page and extract
            # the textual content.
            for page_number, page in enumerate(reader.pages):

                logger.info(f"Parsing page: {page_number + 1}")

                extracted_text : str | None = (page.extract_text())

                # Step 4: Clean the extracted text to remove
                # unnecessary formatting characters.
                cleaned_text : str  = self._clean_text(extracted_text)

                # Step 5: Skip pages that do not contain any
                # meaningful text content.
                if not cleaned_text.strip():

                    logger.warning(f"Skipping empty page: {page_number + 1}")
                    continue
                
                # Step 6: Convert the page content into a
                # LangChain Document and attach metadata for
                # traceability and retrieval.
                document : Document = Document(
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

            # Step 7: Return the parsed document collection.
            return documents

        except Exception as exception:
            logger.exception(f"Failed to parse PDF: {file_path}")
            raise exception
        

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Remove unwanted whitespace and formatting characters
        from extracted PDF text.
        """

        # Return an empty string when no text is available.
        if not text:
            return ""

        # Normalize extracted text by removing line breaks,
        # tab characters, and leading/trailing whitespace.
        return text.replace("\n", " ").replace("\t", " ").strip()