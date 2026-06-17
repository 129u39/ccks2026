"""Stage 9: Self Consistency — 多次推理 + 多数投票"""

from __future__ import annotations

from collections import Counter

from ..llm.client import LLMClient


class SelfConsistency:
    """自一致性模块

    对同一个问题用不同 prompt 风格或温度多次推理，
    取多数答案作为最终结果。
    """

    def __init__(self, llm: LLMClient, n_rounds: int = 3):
        self.llm = llm
        self.n_rounds = n_rounds

    def solve(self, generate_fn, question: str, evidence: str = "") -> str:
        """多次推理 + 投票

        Args:
            generate_fn: 接受 (question, evidence, style_index) 并返回 str 的函数
            question: 问题
            evidence: 证据

        Returns:
            多数投票后的答案
        """
        answers = []
        for i in range(self.n_rounds):
            answer = generate_fn(question, evidence, i)
            answers.append(answer)

        # 投票
        counter = Counter(answers)
        majority = counter.most_common(1)[0][0]
        return majority
