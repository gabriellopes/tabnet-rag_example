# app/src/rag/embeddings.py
import os
from typing import Optional
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from app.src.utils.config_loader import ConfigLoader


class EmbeddingModelLoader:
    """Factory loader for embedding models, configured via app/cfgs/rag.yaml."""

    def __init__(self, model_name: Optional[str] = None):
        rag_cfg = ConfigLoader.load_rag_config()
        embed_cfg = rag_cfg.get("embedding", {})

        # Fallback to YAML config if model_name is not explicitly provided
        self.model_name = model_name or embed_cfg.get("model_name", "models/text-embedding-004")
        
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in environment variables.")

    def load() -> GoogleGenAIEmbedding:
        """Instantiates and returns the configured Google GenAI embedding model."""
        return GoogleGenAIEmbedding(
            model_name=self.model_name,
            api_key=self.api_key,
        )