"""Stage 11: Offline Evaluation Script

Usage:
    python evaluate.py --prediction outputs/submit.jsonl --reference data/contest_data.json
"""

from __future__ import annotations

import argparse
import json

from src.utils.metrics import compute_score


def load_jsonl(path: str) -> list[dict]:
    """加载 JSONL 文件"""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def evaluate(prediction_path: str, label_path: str):
    """离线评测"""
    predictions = load_jsonl(prediction_path)
    pred_map = {p["id"]: p["answer"] for p in predictions}

    # labels 也存为 jsonl 格式 {id, answer} 或直接 json 列表
    with open(label_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        label_map = {item["id"]: item.get("answer", item.get("question", "")) for item in raw}
    else:
        label_map = {raw["id"]: raw.get("answer", raw.get("question", ""))}

    scores = []
    for sid, pred in pred_map.items():
        ref = label_map.get(sid)
        if ref:
            sc = compute_score(pred, ref)
            sc["id"] = sid
            scores.append(sc)

    if not scores:
        print("[Eval] No matching IDs found between prediction and label.")
        return

    avg_em = sum(s["em"] for s in scores) / len(scores)
    avg_f1 = sum(s["f1"] for s in scores) / len(scores)
    avg_rl = sum(s["rouge_l"] for s in scores) / len(scores)
    avg_all = sum(s["avg"] for s in scores) / len(scores)

    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"  Samples:     {len(scores)}")
    print(f"  Exact Match: {avg_em:.4f}")
    print(f"  Token F1:    {avg_f1:.4f}")
    print(f"  ROUGE-L:     {avg_rl:.4f}")
    print(f"  Average:     {avg_all:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Evaluation")
    parser.add_argument("--prediction", required=True, help="Path to prediction JSONL")
    parser.add_argument("--reference", required=True, help="Path to reference JSON")
    args = parser.parse_args()
    evaluate(args.prediction, args.reference)
