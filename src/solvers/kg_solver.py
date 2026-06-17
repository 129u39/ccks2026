"""Stage 5: KG Solver — 知识图谱 QA 主流程"""

from __future__ import annotations

from ..kg.graph_builder import build_graph
from ..kg.graph_reasoner import bfs_search
from ..models import Sample
from ..llm.client import LLMClient


class KGSolver:
    """知识图谱问答求解器

    Workflow:
        Question → Entity Detection → Graph Search → Candidate Answer → LLM Refinement → Final Answer
    """

    def __init__(self, llm: LLMClient, kg_prompt: str = ""):
        self.llm = llm
        self.kg_prompt = kg_prompt

    def solve(self, sample: Sample) -> str:
        """主流程"""
        if not sample.input:
            return self._llm_only(sample.question)

        # 1. 构建图
        G = build_graph(sample.input)

        # 2. 检测问题中的实体
        entities = self._detect_entities(sample.question, G)

        # 3. 图搜索
        evidence = self._search_graph(G, entities)

        # 4. 构造 prompt 并让 LLM 推理
        answer = self._llm_reason(sample.question, evidence)

        return answer

    def _detect_entities(self, question: str, G) -> list[str]:
        """检测问题中提到的实体（简单的字符串匹配）"""
        entities = []
        for node in G.nodes():
            if node.lower() in question.lower():
                entities.append(node)
        return entities

    def _search_graph(self, G, entities: list[str]) -> str:
        """从实体出发 BFS 搜索"""
        lines = []
        for entity in entities:
            results = bfs_search(G, entity, max_hops=2)
            for r in results:
                for u, v, rel in r["path"]:
                    lines.append(f"({u}, {rel}, {v})")
        return "\n".join(lines)

    def _llm_only(self, question: str) -> str:
        """纯 LLM 兜底"""
        return self.llm.generate(question)

    def _llm_reason(self, question: str, evidence: str) -> str:
        """LLM 基于证据推理"""
        if self.kg_prompt:
            prompt = self.kg_prompt.replace("{question}", question).replace("{evidence}", evidence)
        else:
            prompt = (
                f"Based on the knowledge graph triples below, answer the question.\n\n"
                f"Question: {question}\n\n"
                f"Evidence:\n{evidence}\n\n"
                f"Answer:"
            )
        return self.llm.generate(prompt)
