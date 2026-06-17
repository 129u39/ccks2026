"""Stage 6 (Future): Reranker (占位)"""

from __future__ import annotations


class Reranker:
    """精排器（占位）"""

    def rerank(self, query: str, candidates: list[str], topk: int = 3) -> list[str]:
        raise NotImplementedError("Reranker not yet implemented")
