"""Stage 7: Table QA Solver — 基于程序执行的表格问答"""

from __future__ import annotations

import re

import pandas as pd

from ..models import Sample
from ..llm.client import LLMClient
from .table_qa_classifier import TableQuestionClassifier, QuestionType


class TableSolver:
    """表格问答求解器

    Workflow:
        Parse table (header+rows → DataFrame)
        → Classify question (Aggregation / Count / Compare / Lookup)
        → Execute on DataFrame
        → Fallback to LLM if needed
    """

    def __init__(self, llm: LLMClient, table_prompt: str = "", fallback_to_llm: bool = True):
        self.llm = llm
        self.table_prompt = table_prompt
        self.fallback_to_llm = fallback_to_llm
        self.classifier = TableQuestionClassifier()

    def solve(self, sample: Sample) -> str:
        if not sample.table:
            return self._llm_only(sample.question)

        # 1. 解析表格 → DataFrame
        df = self._parse_table(sample.table)
        if df is None or df.empty:
            return self._llm_only(sample.question)

        # 2. 分类问题
        qtype, col = self.classifier.classify(sample.question, df)

        # 3. 执行
        result = self._execute(df, qtype, col, sample.question)
        if result is not None:
            return str(result)

        # 4. 兜底
        if self.fallback_to_llm:
            return self._llm_table_reason(sample.question, df)
        return ""

    def _parse_table(self, table: dict) -> pd.DataFrame | None:
        """将 {header, rows} 转为 DataFrame"""
        try:
            df = pd.DataFrame(table["rows"], columns=table["header"])
            return df
        except Exception:
            return None

    def _execute(self, df: pd.DataFrame, qtype: QuestionType, col: str | None, question: str) -> str | None:
        """在 DataFrame 上执行对应操作"""
        if col and col not in df.columns:
            # 尝试模糊匹配
            for c in df.columns:
                if col.lower() in c.lower() or c.lower() in col.lower():
                    col = c
                    break
            else:
                col = None

        if qtype == QuestionType.AGGREGATION:
            if col and col in df.columns:
                numeric_col = pd.to_numeric(df[col], errors="coerce")
                if "max" in question.lower():
                    return str(numeric_col.max())
                elif "min" in question.lower():
                    return str(numeric_col.min())
                elif "average" in question.lower() or "avg" in question.lower() or "mean" in question.lower():
                    return str(numeric_col.mean())
                elif "sum" in question.lower() or "total" in question.lower():
                    return str(numeric_col.sum())

        elif qtype == QuestionType.COUNT:
            if col and col in df.columns:
                return str(df[col].nunique())
            return str(len(df))

        elif qtype == QuestionType.COMPARE:
            # 比较逻辑：需要更精细的解析，这里简单用 LLM 兜底
            return None

        elif qtype == QuestionType.LOOKUP:
            # 查找逻辑：在表格中搜索问题关键词
            for keyword in re.findall(r"'([^']+)'|\"([^\"]+)\"", question):
                kw = keyword[0] or keyword[1]
                for col in df.columns:
                    mask = df[col].astype(str).str.contains(kw, case=False, na=False)
                    if mask.any():
                        return df[mask].to_string(index=False)
            return None

        return None

    def _llm_only(self, question: str) -> str:
        return self.llm.generate(question)

    def _llm_table_reason(self, question: str, df: pd.DataFrame) -> str:
        """LLM 基于表格推理"""
        table_str = df.to_string(index=False)
        if self.table_prompt:
            prompt = self.table_prompt.replace("{question}", question).replace("{table}", table_str)
        else:
            prompt = (
                f"Answer the question based on the table below.\n\n"
                f"Question: {question}\n\n"
                f"Table:\n{table_str}\n\n"
                f"Answer:"
            )
        return self.llm.generate(prompt)
