from fastapi import APIRouter

from app.ai.retrieval.retrieval_service import RetrievalService

router = APIRouter(prefix="/api/v1/search", tags=["Search"])


@router.get("/")
async def search(query: str, top_k: int = 5):

    retrieval_service: RetrievalService = RetrievalService()

    return await retrieval_service.search(query=query, top_k=top_k)