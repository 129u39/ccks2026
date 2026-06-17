"""Stage 11: Offline Evaluation — 评测指标"""

from __future__ import annotations

from collections import Counter


def normalize(text: str) -> str:
    """标准化文本：小写 + 去除两端空格"""
    return text.strip().lower()


def tokenize(text: str) -> list[str]:
    """简单分词"""
    return normalize(text).split()


def exact_match(prediction: str, reference: str) -> int:
    """Exact Match (EM)"""
    return int(normalize(prediction) == normalize(reference))


def f1_score(prediction: str, reference: str) -> float:
    """Token-level F1"""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L (LCS-based)"""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    # LCS 长度
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_score(prediction: str, reference: str) -> dict[str, float]:
    """计算综合评分 (EM + F1 + ROUGE-L) / 3"""
    em = exact_match(prediction, reference)
    f1 = f1_score(prediction, reference)
    rl = rouge_l(prediction, reference)
    avg = (em + f1 + rl) / 3.0
    return {"em": em, "f1": f1, "rouge_l": rl, "avg": avg}
