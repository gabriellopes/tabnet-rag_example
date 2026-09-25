# app/src/rag/embeddings.py
import os
from typing import Optional
from dotenv import load_dotenv
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from app.src.utils.config_loader import ConfigLoader

load_dotenv()

class EmbeddingModelLoader:
    def __init__(self, model_name: Optional[str] = None):
        rag_cfg = ConfigLoader.load_rag_config()
        embed_cfg = rag_cfg.get("embedding", {})

        raw_name = model_name or embed_cfg.get("model_name", "text-embedding-004")
        
        # Clean the string so LlamaIndex doesn't duplicate the prefix
        self.model_name = raw_name.replace("models/", "").strip("/")

        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in environment variables.")

    def load(self) -> GoogleGenAIEmbedding:
        return GoogleGenAIEmbedding(
            model_name=self.model_name,
            api_key=self.api_key,
        )