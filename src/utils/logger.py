"""Stage 12: Logging — 逐样本保存推理记录"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


class Logger:
    """逐样本日志，便于错误分析"""

    def __init__(self, log_dir: str = "outputs/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"run_{timestamp}.jsonl"
        self._entries = []

    def log(self, sample_id: int, question: str, evidence: str, prediction: str, task_type: str = ""):
        entry = {
            "id": sample_id,
            "task_type": task_type,
            "question": question,
            "evidence": evidence,
            "prediction": prediction,
        }
        self._entries.append(entry)
        # 实时写入
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_all(self) -> list[dict]:
        return self._entries

    def get_path(self) -> str:
        return str(self.log_path)
