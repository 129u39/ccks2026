"""Stage 10 / Stage 2 — 数据加载与提交生成"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import Sample


def load_data(path: str | Path) -> list[Sample]:
    """加载竞赛数据"""
    with open(path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)
    return [Sample.from_dict(d) for d in raw_list]


def load_prompt(path: str | Path) -> str:
    """加载 prompt 模板"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def write_submission(results: list[dict], path: str | Path):
    """生成 submit.jsonl (Stage 10)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 验证
    ids = [r["id"] for r in results]
    assert len(ids) == len(set(ids)), "Duplicate IDs found!"
    for r in results:
        assert isinstance(r["answer"], str), f"Answer for id={r['id']} is not string!"
    print(f"[IO] Submission saved to {path} ({len(results)} samples)")
