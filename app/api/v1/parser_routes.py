from fastapi import APIRouter
from app.ai.parsers.pdf_parser import PDFParser
from app.services.injestion_service import IngestionService

router = APIRouter(prefix="/api/v1/parser", tags=["parser"])


@router.post("/pdf")
async def parse():
    ingestion_service = IngestionService()

    documents = await (
        ingestion_service.parse_file(
            file_path=("C:\Gopal-Projects\Gen-AI\Account Statements\Acct Statement_7803_22042026_20.06.12_unlocked.pdf")
        )
    )

    return {
        "total_documents": len(documents),
        "first_document_metadata":
            documents[0].metadata,
        "first_document_preview":
            documents[0].page_content[:500]
    }