# app/src/plugins/reranker/cross_encoder.py
from typing import Any, List, Dict
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Plugin for reranking vector-retrieved nodes using a Cross-Encoder model."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Any], top_k: int = 3) -> List[Any]:
        """
        Reranks LlamaIndex NodeWithScore objects or strings based on Cross-Encoder scoring.
        """
        if not documents:
            return []

        # Extract text content depending on whether inputs are LlamaIndex Nodes or dicts/strings
        doc_texts = []
        for doc in documents:
            if hasattr(doc, "get_content"):
                doc_texts.append(doc.get_content())
            elif isinstance(doc, dict):
                doc_texts.append(doc.get("text", str(doc)))
            else:
                doc_texts.append(str(doc))

        pairs = [[query, text] for text in doc_texts]
        scores = self.model.predict(pairs)

        # Pair scores with original documents
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k ranked documents
        return [doc for doc, score in scored_docs[:top_k]]