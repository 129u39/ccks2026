"""测试 BM25 检索器"""
import sys; sys.path.insert(0, "..")
from src.retrieval.bm25 import BM25Retriever


def test_bm25_basic():
    r = BM25Retriever()
    r.build_index([
        {"paragraph": "the cat sat on the mat"},
        {"paragraph": "the dog played in the park"},
        {"paragraph": "birds fly in the sky"},
    ])
    results = r.retrieve("cat mat", topk=2)
    assert "cat" in results[0].lower()


def test_bm25_empty_corpus():
    r = BM25Retriever()
    r.build_index([])
    results = r.retrieve("test", topk=5)
    assert results == []


def test_bm25_empty_contexts():
    r = BM25Retriever()
    r.build_index([{"paragraph": ""}, {"paragraph": ""}])
    results = r.retrieve("test", topk=5)
    assert results == []


if __name__ == "__main__":
    test_bm25_basic()
    test_bm25_empty_corpus()
    test_bm25_empty_contexts()
    print("All BM25 tests passed")
