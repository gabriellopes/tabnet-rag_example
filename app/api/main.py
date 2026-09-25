# app/api/main.py
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.src.rag.agents.sus_agent import SUSGraphAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes SUSGraphAgent and automatically opens an Ngrok tunnel on startup."""
    logger.info("Starting up AI-SUS API...")
    
    # --- Start Ngrok Tunnel Automatically ---
    use_ngrok = os.getenv("USE_NGROK", "True").lower() == "true"
    if use_ngrok:
        try:
            from pyngrok import ngrok
            port = os.getenv("PORT", "8000")
            
            # Authtoken setup if present in .env
            authtoken = os.getenv("NGROK_AUTHTOKEN")
            if authtoken:
                ngrok.set_auth_token(authtoken)

            public_url = ngrok.connect(port).public_url
            logger.info("=" * 60)
            logger.info(f"🚀 PUBLIC NGROK URL: {public_url}")
            logger.info("=" * 60)
            app_state["public_url"] = public_url
        except Exception as e:
            logger.warning(f"Could not initialize Ngrok tunnel: {e}")

    try:
        app_state["agent"] = SUSGraphAgent()
        logger.info("SUSGraphAgent successfully initialized.")
        yield
    finally:
        logger.info("Shutting down AI-SUS API...")
        if "public_url" in app_state:
            from pyngrok import ngrok
            ngrok.disconnect(app_state["public_url"])
            ngrok.kill()
        app_state.clear()


app = FastAPI(
    title="AI-SUS Knowledge Graph Agent",
    description="Interoperable GraphRAG API over SINAN, IBGE, CNES, and PBF datasets.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_agent() -> SUSGraphAgent:
    agent = app_state.get("agent")
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent engine is not yet initialized or undergoing maintenance.",
        )
    return agent


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QueryRequest(BaseModel):
    question: str = Field(
        ..., 
        example="Which municipalities present high dengue incidence along with low PBF health compliance?"
    )


class QueryResponse(BaseModel):
    question: str
    answer: str


@app.get("/", tags=["UI"])
def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"service": "AI-SUS GraphRAG API", "status": "online"}


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "service": "AI-SUS GraphRAG",
        "agent_status": "ready" if "agent" in app_state else "uninitialized",
        "public_url": app_state.get("public_url", "Local only")
    }


@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG Agent"])
def query_graph(
    request: QueryRequest, 
    agent: SUSGraphAgent = Depends(get_agent)
):
    try:
        answer = agent.run(request.question)
        return QueryResponse(question=request.question, answer=answer)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the GraphRAG pipeline: {str(e)}",
        )