"""Stage 7: Table Parser — 将 {header, rows} 转为 DataFrame"""

from __future__ import annotations

import pandas as pd


def parse_table(table: dict) -> pd.DataFrame:
    """解析表格数据"""
    return pd.DataFrame(table["rows"], columns=table["header"])
