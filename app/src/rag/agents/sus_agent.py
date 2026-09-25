# app/src/rag/agents/agent.py
import logging
from typing import List, Dict, Any

from app.src.utils.config_loader import ConfigLoader
from app.src.rag.retriever import GraphRAGRetriever
from app.src.rag.generation import GeminiGenerator
from app.src.rag.agents.tools import GraphTools
from app.src.plugins.reranker.mmr import MMRReranker
from app.src.plugins.reranker.cohere import CohereReranker
from app.src.plugins.reranker.cross_encoder import CrossEncoderReranker

logger = logging.getLogger(__name__)


class SUSGraphAgent:
    """Core Orchestrator agent integrating GraphRAG, dynamic Reranking, and Gemini Generation."""

    def __init__(self):
        logger.info("Initializing SUS GraphRAG Agent...")
        self.rag_cfg = ConfigLoader.load_rag_config()
        self.reranker_cfg = ConfigLoader.load_reranker_config()

        # Components
        self.retriever = GraphRAGRetriever()
        self.tools = GraphTools()
        self.generator = GeminiGenerator()
        self.reranker = self._load_reranker_plugin()

    def _load_reranker_plugin(self):
        active = self.reranker_cfg.get("active_plugin", "mmr")
        logger.info(f"Activating Reranker Plugin: {active}")

        if active == "mmr":
            cfg = self.reranker_cfg.get("mmr", {})
            return MMRReranker(lambda_mult=cfg.get("lambda_mult", 0.5))
        elif active == "cohere":
            cfg = self.reranker_cfg.get("cohere", {})
            return CohereReranker(model=cfg.get("model", "rerank-v3.5"))
        elif active == "cross_encoder":
            cfg = self.reranker_cfg.get("cross_encoder", {})
            return CrossEncoderReranker(model_name=cfg.get("model_name", "BAAI/bge-reranker-base"))
        else:
            raise ValueError(f"Unknown reranker plugin: {active}")

    def run(self, query: str) -> str:
        """Full Execution Loop: Retrieve -> Rerank -> Generate."""
        # 1. Retrieve raw candidates
        top_k_retrieve = self.rag_cfg.get("retrieval", {}).get("top_k", 10)
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k_retrieve)

        # 2. Rerank context using active plugin
        active_plugin = self.reranker_cfg.get("active_plugin", "mmr")
        top_k_rerank = self.reranker_cfg.get(active_plugin, {}).get("top_k", 3)
        reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=top_k_rerank)

        # 3. Generate answer via Gemini
        return self.generator.generate_response(query, reranked_docs)