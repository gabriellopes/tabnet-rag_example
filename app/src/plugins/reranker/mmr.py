# app/src/plugins/reranker/mmr_reranker.py
from typing import Any, List
import numpy as np

from app.src.plugins.reranker.base import BaseReranker
from app.src.rag.embeddings import EmbeddingModelLoader


class MMRReranker(BaseReranker):
    """Plugin for Maximal Marginal Relevance (MMR) reranking to maximize diversity and relevance."""

    def __init__(self, lambda_mult: float = 0.5):
        """
        :param lambda_mult: Controls balance between relevance (1.0) and diversity (0.0).
        """
        self.lambda_mult = lambda_mult
        self.embed_model = EmbeddingModelLoader().load()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def rerank(self, query: str, documents: List[Any], top_k: int = 3) -> List[Any]:
        """Selects top_k documents using MMR logic over vector representations."""
        if not documents:
            return []

        doc_texts = []
        for doc in documents:
            if hasattr(doc, "get_content"):
                doc_texts.append(doc.get_content())
            elif isinstance(doc, dict):
                doc_texts.append(doc.get("text", str(doc)))
            else:
                doc_texts.append(str(doc))

        # Get embeddings for query and candidate documents
        query_embedding = np.array(self.embed_model.get_query_embedding(query))
        doc_embeddings = [
            np.array(self.embed_model.get_text_embedding(text)) for text in doc_texts
        ]

        # Calculate relevance scores to query
        query_sims = [
            self._cosine_similarity(query_embedding, doc_emb)
            for doc_emb in doc_embeddings
        ]

        selected_indices: List[int] = []
        unselected_indices = list(range(len(documents)))

        while len(selected_indices) < min(top_k, len(documents)):
            best_score = -float("inf")
            best_idx = -1

            for idx in unselected_indices:
                rel_score = query_sims[idx]

                # Compute maximum similarity with already selected items
                if not selected_indices:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        self._cosine_similarity(
                            doc_embeddings[idx], doc_embeddings[s_idx]
                        )
                        for s_idx in selected_indices
                    )

                # MMR equation: λ * Relevance - (1 - λ) * Redundancy
                mmr_score = (self.lambda_mult * rel_score) - (
                    (1 - self.lambda_mult) * redundancy
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)

        return [documents[i] for i in selected_indices]