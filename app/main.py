from fastapi import FastAPI

from app.api.v1.parser_routes import router as parser_router
from app.api.v1.retrieval_router import router as retrieval_router

app = FastAPI(title="AI Expense Analyzer", version="1.0.0")


@app.get("/")
async def health_check():

    return {
        "status": "Running"
    }


app.include_router(parser_router)
app.include_router(retrieval_router)