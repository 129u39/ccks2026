"""Stage 8: Evidence Verification — 验证答案是否被证据充分支持"""

from __future__ import annotations

from ..llm.client import LLMClient


class AnswerVerifier:
    """答案验证器

    检查 LLM 生成的答案是否被提供的证据充分支持。
    如果答案无据可依，返回 "Unknown"。
    """

    def __init__(self, llm: LLMClient, verify_prompt: str = ""):
        self.llm = llm
        self.verify_prompt = verify_prompt

    def verify(self, question: str, evidence: str, answer: str) -> str:
        """验证答案

        Returns:
            原始 answer（若验证通过）或 "Unknown"（若未通过）
        """
        if self.verify_prompt:
            prompt = self.verify_prompt.replace("{question}", question)
            prompt = prompt.replace("{evidence}", evidence)
            prompt = prompt.replace("{answer}", answer)
        else:
            prompt = (
                f"Question: {question}\n\n"
                f"Evidence:\n{evidence}\n\n"
                f"Answer: {answer}\n\n"
                f"Is the answer fully supported by the evidence?\n"
                f"Return YES or NO."
            )

        verdict = self.llm.generate(prompt).strip().upper()

        if "YES" in verdict:
            return answer
        else:
            return "Unknown"
