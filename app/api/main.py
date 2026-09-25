# app/api/main.py
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field

from app.src.rag.agents.sus_agent import SUSGraphAgent

logger = logging.getLogger("API")

# Global container for application resources
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan context manager.
    Handles startup state initialization (e.g., loading vector stores, models, YAML configs)
    and handles graceful shutdown teardown.
    """
    logger.info("Starting up AI-SUS API... Initializing SUSGraphAgent.")
    try:
        # Initialize the single agent instance during startup
        app_state["agent"] = SUSGraphAgent()
        logger.info("SUSGraphAgent successfully initialized and ready for queries.")
        yield
    finally:
        logger.info("Shutting down AI-SUS API... Cleaning up app state resources.")
        app_state.clear()


app = FastAPI(
    title="AI-SUS Knowledge Graph Agent",
    description="Interoperable GraphRAG API over SINAN, IBGE, CNES, and PBF datasets.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_agent() -> SUSGraphAgent:
    """Dependency Provider for accessing the singleton SUSGraphAgent instance."""
    agent = app_state.get("agent")
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent engine is not yet initialized or undergoing maintenance.",
        )
    return agent


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    question: str = Field(
        ..., 
        example="Which municipalities present high dengue incidence along with low PBF health compliance?"
    )


class QueryResponse(BaseModel):
    question: str
    answer: str


# --- Endpoints ---

@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint to verify service operational status."""
    return {
        "status": "ok",
        "service": "AI-SUS GraphRAG",
        "agent_status": "ready" if "agent" in app_state else "uninitialized",
    }


@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG Agent"])
def query_graph(
    request: QueryRequest, 
    agent: SUSGraphAgent = Depends(get_agent)
):
    """
    Receives a natural language question, triggers retrieval over the GraphRAG index,
    applies active Reranking (e.g. MMR/Cohere), and generates an answer via Gemini.
    """
    try:
        answer = agent.run(request.question)
        return QueryResponse(question=request.question, answer=answer)
    except Exception as e:
        logger.error(f"Error processing question '{request.question}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the GraphRAG pipeline: {str(e)}",
        )