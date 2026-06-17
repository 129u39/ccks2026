"""测试统一数据模型"""
import sys; sys.path.insert(0, "..")
from src.models import Sample


def test_sample_from_dict():
    d = {"id": 1, "task_type": "knowledge_graph", "question": "test?", "input": "triples..."}
    s = Sample.from_dict(d)
    assert s.id == 1
    assert s.task_type == "knowledge_graph"
    assert s.question == "test?"
    assert s.input == "triples..."
    assert s.contexts == []


def test_sample_from_dict_full():
    d = {
        "id": 2,
        "task_type": "multi_hop_qa",
        "question": "q?",
        "contexts": [{"title": "T1", "sentences": ["s1"], "paragraph": "p1"}],
    }
    s = Sample.from_dict(d)
    assert s.id == 2
    assert len(s.contexts) == 1


def test_sample_from_dict_table():
    d = {"id": 3, "task_type": "table_qa", "question": "q?", "table": {"header": ["A"], "rows": [["1"]]}}
    s = Sample.from_dict(d)
    assert s.table["header"] == ["A"]


if __name__ == "__main__":
    test_sample_from_dict()
    test_sample_from_dict_full()
    test_sample_from_dict_table()
    print("All model tests passed")
