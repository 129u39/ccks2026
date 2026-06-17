"""Stage 7: Table Executor — 在 DataFrame 上执行程序化操作"""

from __future__ import annotations

import pandas as pd


def execute_aggregation(df: pd.DataFrame, column: str, op: str) -> float:
    """执行聚合操作"""
    col = pd.to_numeric(df[column], errors="coerce")
    if op == "max":
        return col.max()
    elif op == "min":
        return col.min()
    elif op == "avg":
        return col.mean()
    elif op == "sum":
        return col.sum()
    raise ValueError(f"Unknown aggregation: {op}")


def execute_count(df: pd.DataFrame, column: str | None = None) -> int:
    """执行计数"""
    if column:
        return df[column].nunique()
    return len(df)


def execute_lookup(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """执行查找"""
    return df[df[column].astype(str).str.contains(value, case=False, na=False)]
