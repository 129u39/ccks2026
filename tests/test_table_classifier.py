"""测试 Table Question Classifier"""
import sys; sys.path.insert(0, "..")
import pandas as pd
from src.solvers.table_qa_classifier import TableQuestionClassifier, QuestionType


def test_classify_aggregation():
    clf = TableQuestionClassifier()
    df = pd.DataFrame({"Score": [10, 20, 30]})
    qtype, col = clf.classify("What is the maximum Score?", df)
    assert qtype == QuestionType.AGGREGATION
    assert col == "Score"


def test_classify_count():
    clf = TableQuestionClassifier()
    df = pd.DataFrame({"Name": ["A", "B", "C"]})
    qtype, col = clf.classify("How many names are there?", df)
    assert qtype == QuestionType.COUNT


def test_classify_lookup():
    clf = TableQuestionClassifier()
    df = pd.DataFrame({"Venue": ["A", "B"]})
    qtype, col = clf.classify("Which venue is used?", df)
    assert qtype == QuestionType.LOOKUP


def test_classify_compare():
    clf = TableQuestionClassifier()
    df = pd.DataFrame({"Score": [10, 20]})
    qtype, col = clf.classify("Score greater than 15?", df)
    assert qtype == QuestionType.COMPARE


def test_classify_unknown():
    clf = TableQuestionClassifier()
    df = pd.DataFrame({"A": [1]})
    qtype, col = clf.classify("Some random question?", df)
    assert qtype == QuestionType.UNKNOWN


if __name__ == "__main__":
    test_classify_aggregation()
    test_classify_count()
    test_classify_lookup()
    test_classify_compare()
    test_classify_unknown()
    print("All table classifier tests passed")
