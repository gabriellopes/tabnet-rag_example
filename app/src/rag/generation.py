# app/src/rag/generation.py
import os
from typing import List, Any
from llama_index.llms.google_genai import GoogleGenAI
from app.src.utils.config_loader import ConfigLoader


class GeminiGenerator:
    """Generation module utilizing Google Gemini models via LlamaIndex integration."""

    def __init__(self):
        rag_cfg = ConfigLoader.load_rag_config()
        gen_cfg = rag_cfg.get("generation", {})

        self.model_name = gen_cfg.get("model", "gemini-2.5-flash")
        self.temperature = gen_cfg.get("temperature", 0.2)

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set."
            )

        self.llm = GoogleGenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
        )

    def generate_response(self, query: str, context_docs: List[Any]) -> str:
        """Synthesizes a response given the user query and reranked context documents."""
        context_str = "\n\n".join(
            [
                doc.get_content() if hasattr(doc, "get_content") else str(doc)
                for doc in context_docs
            ]
        )

        prompt = f"""
You are an expert AI public health and data analyst specializing in Brazil's SUS and MDS datasets.
Answer the user's question using only the retrieved Knowledge Graph context provided below.

Context:
{context_str}

Question: {query}

Answer:
"""
        response = self.llm.complete(prompt)
        return response.text