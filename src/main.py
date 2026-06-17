"""CCKS2026 Complex Knowledge Reasoning Challenge — 主入口"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .models import Sample
from .router import Router
from .llm.client import LLMClient, LLMConfig
from .solvers.kg_solver import KGSolver
from .solvers.text_solver import TextSolver
from .solvers.table_solver import TableSolver
from .utils.io import load_data, load_prompt, write_submission
from .utils.logger import Logger
from .utils.metrics import compute_score


def main(
    data_path: str = "data/contest_data.json",
    output_path: str = "outputs/submit.jsonl",
    prompt_dir: str = "prompts",
    model_name: str = "qwen-max",
    provider: str = "openai",
    api_key: str = "",
    temperature: float = 0.1,
    bm25_topk: int = 5,
    log_dir: str = "outputs/logs",
    label_path: str = "",
):
    # 1. 加载数据
    samples = load_data(data_path)
    print(f"[Main] Loaded {len(samples)} samples")

    # 2. 初始化 LLM
    config = LLMConfig(
        provider=provider,
        model_name=model_name,
        api_key=api_key or os.environ.get("API_KEY", ""),
        temperature=temperature,
    )
    llm = LLMClient(config)

    # 3. 加载 Prompt
    kg_prompt = ""
    text_prompt = ""
    table_prompt = ""
    verify_prompt = ""
    if os.path.exists(prompt_dir):
        for fname in ["kg_prompt.txt", "text_prompt.txt", "table_prompt.txt", "verify_prompt.txt"]:
            fpath = os.path.join(prompt_dir, fname)
            if os.path.exists(fpath):
                content = load_prompt(fpath)
                if "kg_prompt" in fname:
                    kg_prompt = content
                elif "text_prompt" in fname:
                    text_prompt = content
                elif "table_prompt" in fname:
                    table_prompt = content
                elif "verify_prompt" in fname:
                    verify_prompt = content

    # 4. 初始化 Solver
    kg_solver = KGSolver(llm=llm, kg_prompt=kg_prompt)
    text_solver = TextSolver(llm=llm, text_prompt=text_prompt, topk=bm25_topk)
    table_solver = TableSolver(llm=llm, table_prompt=table_prompt, fallback_to_llm=True)

    # 5. 路由器
    router = Router(
        kg_solver=kg_solver,
        text_solver=text_solver,
        table_solver=table_solver,
    )

    # 6. 日志
    logger = Logger(log_dir=log_dir)

    # 7. 执行推理
    results = []
    for i, sample in enumerate(samples):
        print(f"[Main] [{i+1}/{len(samples)}] id={sample.id} type={sample.task_type}")

        answer = router.route(sample)

        # 记录日志
        logger.log(
            sample_id=sample.id,
            question=sample.question,
            evidence=str(sample.contexts or sample.input or sample.table or ""),
            prediction=answer,
            task_type=sample.task_type,
        )

        results.append({"id": sample.id, "answer": answer})

    # 8. 生成提交文件
    write_submission(results, output_path)

    # 9. 如果有标签，离线评估
    if label_path and os.path.exists(label_path):
        labels = load_data(label_path)
        label_map = {s.id: s.question for s in labels}  # 用 question 做 reference
        # 实际竞赛中 labels 应有 answer 字段；这里用 question 占位
        scores = []
        for r in results:
            ref = label_map.get(r["id"], "")
            if ref:
                score = compute_score(r["answer"], ref)
                scores.append(score)
        if scores:
            avg_em = sum(s["em"] for s in scores) / len(scores)
            avg_f1 = sum(s["f1"] for s in scores) / len(scores)
            avg_rl = sum(s["rouge_l"] for s in scores) / len(scores)
            avg_all = sum(s["avg"] for s in scores) / len(scores)
            print(f"[Eval] EM={avg_em:.4f} F1={avg_f1:.4f} ROUGE-L={avg_rl:.4f} AVG={avg_all:.4f}")

    print(f"[Main] Done. Output: {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCKS2026 Complex Knowledge Reasoning Challenge")
    parser.add_argument("--data_path", default="data/contest_data.json")
    parser.add_argument("--output_path", default="outputs/submit.jsonl")
    parser.add_argument("--prompt_dir", default="prompts")
    parser.add_argument("--model_name", default="qwen-max")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--bm25_topk", type=int, default=5)
    parser.add_argument("--log_dir", default="outputs/logs")
    parser.add_argument("--label_path", default="")
    args = parser.parse_args()

    main(**vars(args))
