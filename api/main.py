from fastapi import FastAPI
from pydantic import BaseModel

from rag.query_engine import RAGEngine


app = FastAPI(
    title="AeroDocs RAG API",
    version="1.0"
)

# Load once when API starts
rag_engine = RAGEngine()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/api/v1/query",
          response_model=QueryResponse
          )
def query(request: QueryRequest):
    result = rag_engine.query(request.question)
    return result
