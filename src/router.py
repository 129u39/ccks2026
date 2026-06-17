"""Stage 3: Task Router — 根据 task_type 分发到对应 Solver"""

from __future__ import annotations

from .models import Sample
from .solvers.kg_solver import KGSolver
from .solvers.text_solver import TextSolver
from .solvers.table_solver import TableSolver


class Router:
    """任务路由器"""

    def __init__(self, kg_solver: KGSolver, text_solver: TextSolver, table_solver: TableSolver):
        self.kg_solver = kg_solver
        self.text_solver = text_solver
        self.table_solver = table_solver

    def route(self, sample: Sample) -> str:
        """根据 task_type 分发到对应求解器"""
        if sample.task_type == "knowledge_graph":
            return self.kg_solver.solve(sample)
        elif sample.task_type == "multi_hop_qa":
            return self.text_solver.solve(sample)
        elif sample.task_type == "table_qa":
            return self.table_solver.solve(sample)
        else:
            raise ValueError(f"Unknown task_type: {sample.task_type}")
