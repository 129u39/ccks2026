"""Stage 7: Table Question Classifier — 问题分类器"""

from __future__ import annotations

from enum import Enum

import pandas as pd


class QuestionType(Enum):
    AGGREGATION = "aggregation"  # max / min / avg / sum
    COUNT = "count"             # how many / count
    COMPARE = "compare"         # greater than / less than
    LOOKUP = "lookup"           # who / which / what
    UNKNOWN = "unknown"


class TableQuestionClassifier:
    """基于规则的问题分类"""

    def classify(self, question: str, df: pd.DataFrame) -> tuple[QuestionType, str | None]:
        q = question.lower()

        # 尝试识别目标列
        col = self._detect_column(question, df)

        # Count
        if any(word in q for word in ["how many", "count ", "number of", "distinct"]):
            return QuestionType.COUNT, col

        # Aggregation
        if any(word in q for word in ["maximum", "max ", "largest", "highest", "most"]):
            return QuestionType.AGGREGATION, col
        if any(word in q for word in ["minimum", "min ", "smallest", "lowest", "least"]):
            return QuestionType.AGGREGATION, col
        if any(word in q for word in ["average", "avg ", "mean "]):
            return QuestionType.AGGREGATION, col
        if any(word in q for word in ["sum ", "total "]):
            return QuestionType.AGGREGATION, col

        # Compare
        if any(word in q for word in ["greater than", "more than", "larger than", ">", "less than", "fewer than", "<", "equal to", "same as"]):
            return QuestionType.COMPARE, col

        # Lookup
        if any(word in q for word in ["who", "which", "what", "when", "where", "name", "list", "find", "show"]):
            return QuestionType.LOOKUP, col

        return QuestionType.UNKNOWN, col

    def _detect_column(self, question: str, df: pd.DataFrame) -> str | None:
        """检测问题中提到的列名"""
        q = question.lower()
        for col in df.columns:
            if col.lower() in q:
                return col
        # 别名匹配
        alias_map = {
            "score": ["Score", "score", "结果", "比分"],
            "name": ["Name", "name", "名称", "姓名"],
            "date": ["Date", "date", "日期"],
            "venue": ["Venue", "venue", "场地"],
            "count": ["Count", "count"],
        }
        for alias, candidates in alias_map.items():
            for c in candidates:
                if c.lower() in q:
                    for df_col in df.columns:
                        if alias in df_col.lower() or df_col.lower() in alias:
                            return df_col
        return None
