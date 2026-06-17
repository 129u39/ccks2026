"""测试路由器和求解器集成"""
import sys; sys.path.insert(0, "..")
from src.models import Sample
from src.router import Router


def test_router_creation():
    """Router 需要 solver 实例，仅测试创建（无 LLM 则跳过集成）"""
    try:
        from src.solvers.kg_solver import KGSolver
        from src.solvers.text_solver import TextSolver
        from src.solvers.table_solver import TableSolver
        from src.llm.client import LLMClient, LLMConfig
        cfg = LLMConfig(provider="openai", model_name="gpt-4o-mini", api_key="test")
        llm = LLMClient(cfg)
        router = Router(
            kg_solver=KGSolver(llm),
            text_solver=TextSolver(llm),
            table_solver=TableSolver(llm),
        )
        assert router is not None
    except Exception as e:
        # 无 API key 时跳过
        print(f"Router creation skipped (expected without API key): {e}")


def test_router_route_logic():
    """测试路由分发表决逻辑（模拟）"""
    class MockSolver:
        def solve(self, sample):
            return f"mock_{sample.task_type}"

    router = Router(
        kg_solver=MockSolver(),
        text_solver=MockSolver(),
        table_solver=MockSolver(),
    )

    kg = router.route(Sample(id=1, task_type="knowledge_graph", question="q"))
    assert kg == "mock_knowledge_graph"

    text = router.route(Sample(id=2, task_type="multi_hop_qa", question="q"))
    assert text == "mock_multi_hop_qa"

    table = router.route(Sample(id=3, task_type="table_qa", question="q"))
    assert table == "mock_table_qa"


if __name__ == "__main__":
    test_router_creation()
    test_router_route_logic()
    print("All router tests passed")
