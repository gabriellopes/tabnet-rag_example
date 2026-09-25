# app/src/plugins/reranker/base.py
from abc import ABC, abstractmethod
from typing import Any, List


class BaseReranker(ABC):
    """Abstract Base Class for all reranker plugins."""

    @abstractmethod
    def rerank(self, query: str, documents: List[Any], top_k: int = 3) -> List[Any]:
        """
        Reranks a list of retrieved documents or nodes relative to a user query.

        :param query: Natural language user query.
        :param documents: List of retrieved context objects (strings, dicts, or LlamaIndex Nodes).
        :param top_k: Number of top reranked items to return.
        :return: Reranked subset of documents.
        """
        pass