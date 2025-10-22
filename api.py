import logging
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG FAISS Server API",
    version="1.0.0",
    description="FAISS Vector Database API for Document Retrieval"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = VectorStore()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any] = {}
    rank: int

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int


@app.get("/health")
async def health():
    if vector_store.load_index():
        return {"status": "ok"}
    raise HTTPException(status_code=500, detail="Vector store not available")


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    try:
        results = vector_store.get_vector_results(request.query, top_k=request.top_k)
        if not results:
            raise HTTPException(status_code=404, detail="No results found")

        return SearchResponse(
            query=request.query,
            results=[SearchResult(**res) for res in results],
            total_found=len(results)
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search operation failed")

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="api:app", host="localhost", port=8000)