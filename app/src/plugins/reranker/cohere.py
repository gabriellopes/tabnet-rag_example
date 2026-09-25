# app/src/plugins/reranker/cohere_reranker.py
import os
import logging
from typing import Any, List, Optional
import cohere

from app.src.plugins.reranker.base import BaseReranker

logger = logging.getLogger(__name__)


class CohereReranker(BaseReranker):
    """Plugin for reranking documents using Cohere's dedicated Rerank API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-v3.5",
    ):
        self.model = model
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.client = None

        if self.api_key:
            self.client = cohere.ClientV2(api_key=self.api_key)
        else:
            logger.warning(
                "[CohereReranker] COHERE_API_KEY not found. "
                "Reranker will operate in fallback mode (returning top_k un-reranked)."
            )

    def rerank(self, query: str, documents: List[Any], top_k: int = 3) -> List[Any]:
        """Reranks retrieved documents using Cohere API, or falls back gracefully if key is missing."""
        if not documents:
            return []

        # Fallback mechanism if API key was not supplied
        if not self.client:
            logger.info("[CohereReranker] Bypassing Cohere API call; returning top_k original docs.")
            return documents[:top_k]

        doc_texts = []
        for doc in documents:
            if hasattr(doc, "get_content"):
                doc_texts.append(doc.get_content())
            elif isinstance(doc, dict):
                doc_texts.append(doc.get("text", str(doc)))
            else:
                doc_texts.append(str(doc))

        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=top_k,
            )
            reranked_docs = [documents[result.index] for result in response.results]
            return reranked_docs
        except Exception as e:
            logger.error(f"[CohereReranker] API call failed with error: {e}. Falling back to default top_k.")
            return documents[:top_k]