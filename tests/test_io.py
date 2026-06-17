"""测试 I/O 工具"""
import sys; sys.path.insert(0, "..")
import json
import tempfile
from pathlib import Path
from src.utils.io import write_submission, load_prompt


def test_write_submission():
    results = [{"id": 1, "answer": "yes"}, {"id": 2, "answer": "no"}]
    tmp = Path(tempfile.gettempdir()) / "test_submit.jsonl"
    write_submission(results, tmp)
    assert tmp.exists()
    with open(tmp, "r") as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["answer"] == "yes"
    tmp.unlink()


def test_write_submission_validation():
    """测试验证：answer 必须是 string"""
    results = [{"id": 1, "answer": "42"}]  # string is fine
    tmp = Path(tempfile.gettempdir()) / "test_submit2.jsonl"
    write_submission(results, tmp)
    tmp.unlink()
    # 如果传 int 会断言失败
    try:
        write_submission([{"id": 1, "answer": 42}], tmp)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_load_prompt():
    tmp = Path(tempfile.gettempdir()) / "test_prompt.txt"
    tmp.write_text("Answer: {question}", encoding="utf-8")
    content = load_prompt(tmp)
    assert "Answer:" in content
    tmp.unlink()


if __name__ == "__main__":
    test_write_submission()
    test_write_submission_validation()
    test_load_prompt()
    print("All IO tests passed")
