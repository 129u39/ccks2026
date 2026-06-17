"""Stage 6: Multi-Hop Text QA Solver — 检索 + LLM 推理"""

from __future__ import annotations

from ..models import Sample
from ..llm.client import LLMClient
from ..retrieval.bm25 import BM25Retriever


class TextSolver:
    """多跳文本问答求解器

    Workflow:
        1. Build paragraph corpus from contexts
        2. BM25 retrieve Top-K
        3. Construct reasoning context
        4. LLM reasoning with evidence
    """

    def __init__(self, llm: LLMClient, text_prompt: str = "", topk: int = 5):
        self.llm = llm
        self.text_prompt = text_prompt
        self.topk = topk
        self.retriever = BM25Retriever()

    def solve(self, sample: Sample) -> str:
        """主流程"""
        # 1. 构建段落语料
        self.retriever.build_index(sample.contexts)

        # 2. BM25 检索
        top_paragraphs = self.retriever.retrieve(sample.question, topk=self.topk)

        # 3. 构造上下文
        evidence = "\n\n".join(top_paragraphs)

        # 4. LLM 推理
        if self.text_prompt:
            prompt = self.text_prompt.replace("{question}", sample.question).replace("{evidence}", evidence)
        else:
            prompt = (
                f"Use ONLY the provided evidence to answer the question.\n\n"
                f"Question: {sample.question}\n\n"
                f"Evidence:\n{evidence}\n\n"
                f"Answer:"
            )

        return self.llm.generate(prompt)
