"""测试评测指标"""
import sys; sys.path.insert(0, "..")
from src.utils.metrics import exact_match, f1_score, rouge_l, compute_score


def test_exact_match():
    assert exact_match("hello", "hello") == 1
    assert exact_match("Hello", "hello") == 1
    assert exact_match("hello", "world") == 0


def test_f1_score():
    assert f1_score("a b c", "a b c") == 1.0
    assert f1_score("a b c", "a b") > 0
    assert f1_score("a b c", "d e f") == 0.0


def test_rouge_l():
    assert rouge_l("a b c", "a b c") == 1.0
    assert rouge_l("a b c", "a b") > 0.5
    assert rouge_l("a b c", "d e f") == 0.0


def test_compute_score():
    sc = compute_score("hello world", "hello world")
    assert sc["em"] == 1.0
    assert sc["avg"] == 1.0
    sc2 = compute_score("hello", "world")
    assert sc2["em"] == 0.0


if __name__ == "__main__":
    test_exact_match()
    test_f1_score()
    test_rouge_l()
    test_compute_score()
    print("All metric tests passed")
