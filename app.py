from fastapi import FastAPI,HTTPException
from contextlib import asynccontextmanager
from retriever import GoogleRAGRetriever 
import os
from typing import Optional
import logging
from pydantic import BaseModel
import asyncio

required_vars = ["GOOGLE_API_KEY", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_NAME"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required env var: {var}")

logging.basicConfig(
    level=logging.DEBUG,  # Set to INFO or ERROR in production
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 3
    similarity_threshold: Optional[float] = 0.2
    include_sources: Optional[bool] = True # For Testing ON

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CONNECTION_STRING = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}"
    app.state.retriever = GoogleRAGRetriever(
        connection_string=CONNECTION_STRING,
        google_api_key=GOOGLE_API_KEY,
        collection_name="faq_embeddings",
        llm_model=os.getenv("GOOGLE_LLM_MODEL") or "models/gemini-2.0-flash"
    )
    yield
    #shutdown

app = FastAPI(title="Google RAG Retriever API", version="1.0",lifespan=lifespan)

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    retriever = app.state.retriever
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    try:
        result = await asyncio.to_thread(
            retriever.query,
            q=req.question,
            k=req.k,
            similarity_threshold=req.similarity_threshold,
            include_sources=req.include_sources
        )
        logger.debug(f"Query result: {result}")
        return {"answer": result['answer']}
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/stats")
def get_stats():
    try:
        return app.state.retriever.get_collection_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))