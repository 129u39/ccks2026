"""Stage 6 (Future): Dense Embedding Retriever"""

from __future__ import annotations


class EmbeddingRetriever:
    """稠密检索（占位）"""

    def __init__(self, model_name: str = "BAAI/bge-large-zh"):
        self.model_name = model_name
        self._model = None

    def build_index(self, paragraphs: list[str]):
        raise NotImplementedError("Dense retriever not yet implemented")

    def retrieve(self, query: str, topk: int = 5) -> list[str]:
        raise NotImplementedError("Dense retriever not yet implemented")
