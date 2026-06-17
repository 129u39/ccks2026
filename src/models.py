"""Stage 2: Unified Data Schema"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    """统一样本对象，屏蔽不同数据集的差异"""
    id: int
    task_type: str  # knowledge_graph | multi_hop_qa | table_qa
    question: str

    # 知识图谱输入
    input: str | None = None

    # 多跳文本 QA 输入
    contexts: list[dict] | None = field(default_factory=list)

    # 表格 QA 输入
    table: dict | None = None

    # 原始数据（保留以便调试 / 扩展）
    raw: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(
            id=d["id"],
            task_type=d["task_type"],
            question=d["question"],
            input=d.get("input"),
            contexts=d.get("contexts") or [],
            table=d.get("table"),
            raw=d,
        )
