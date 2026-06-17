"""Stage 6: BM25 Retriever — 稀疏检索"""

from __future__ import annotations

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 检索器"""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.paragraphs: list[str] = []

    def build_index(self, contexts: list[dict]):
        """从 contexts 构建 BM25 索引"""
        self.paragraphs = []
        for ctx in contexts:
            para = ctx.get("paragraph", "")
            if para:
                self.paragraphs.append(para)

        if self.paragraphs:
            tokenized = [self._tokenize(p) for p in self.paragraphs]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def retrieve(self, query: str, topk: int = 5) -> list[str]:
        """检索 Top-K 段落"""
        if not self.bm25 or not self.paragraphs:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]

        return [self.paragraphs[i] for i in top_indices if scores[i] > 0]

    def _tokenize(self, text: str) -> list[str]:
        """简单分词"""
        return text.lower().split()
