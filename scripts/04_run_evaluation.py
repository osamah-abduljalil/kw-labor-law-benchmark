#!/usr/bin/env python3
"""CLI Script 04: Run automated LLM-as-a-judge evaluation (single answer or batch CSV)."""

import argparse
from pathlib import Path
import sys

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

from config import load_config
from evaluation.judge import LegalBenchmarkJudge
from models.api_models import DeepSeekLLM, GeminiLLM, OpenAILLM


def get_evaluator_llm(model_name: str):
    name_lower = model_name.lower()
    if "deepseek" in name_lower:
        return DeepSeekLLM(model_name=model_name)
    elif "gemini" in name_lower:
        return GeminiLLM(model_name=model_name)
    else:
        return OpenAILLM(model_name=model_name)


def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge automated evaluation (single Q&A or batch CSV).")
    parser.add_argument("--config", "-c", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--answers", "-a", type=str, default=None, help="Path to CSV file with generated model answers")
    parser.add_argument("--question", "-q", type=str, default=None, help="Evaluate a single legal question string")
    parser.add_argument("--answer", type=str, default=None, help="Evaluate a single model answer string")
    parser.add_argument("--output", "-o", type=str, default="evaluations/generated_evaluation_results.csv", help="Output evaluation CSV")
    parser.add_argument("--evaluator", "-e", type=str, default=None, help="Evaluator model name (e.g. deepseek-chat, gemini-2.5-flash)")
    parser.add_argument("--metrics", "-m", nargs="+", default=["completeness", "correctness", "fluency"], help="Metrics to evaluate")

    args = parser.parse_args()
    cfg = load_config(args.config)

    evaluator_name = args.evaluator or cfg.evaluation.evaluator_model
    print(f"Initializing evaluator model: {evaluator_name}")
    evaluator_llm = get_evaluator_llm(evaluator_name)
    judge = LegalBenchmarkJudge(evaluator_llm)

    # --- Mode 1: Evaluate a single answer directly ---
    if args.question and args.answer:
        print(f"\nEvaluating single response across metrics: {args.metrics}")
        for metric in args.metrics:
            res = judge.evaluate_metric(args.question, args.answer, metric)
            print(f"\n[{metric.upper()}] Score: {res.get('score')}/5")
            print(f"Justification: {res.get('justification')}")
        return

    # --- Mode 2: Batch CSV evaluation ---
    if not args.answers:
        # Check default locations
        candidates = [
            "data/generated/rag_answers_all_evd_score.csv",
            "data/generated/rag_benchmark_answers.csv",
        ]
        for c in candidates:
            if Path(c).exists():
                args.answers = c
                break

    if not args.answers or not Path(args.answers).exists():
        print("Error: Please provide an answers CSV file via --answers or evaluate a single answer using --question and --answer.", file=sys.stderr)
        sys.exit(1)

    if pd is None:
        print("Error: pandas is required for batch CSV evaluation. Please install pandas.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading answers from: {args.answers}")
    df_answers = pd.read_csv(args.answers)
    answer_cols = [c for c in df_answers.columns if c.startswith("answer_")]
    print(f"Found {len(answer_cols)} answer columns to evaluate: {answer_cols}")

    results = []
    for idx, row in tqdm(df_answers.iterrows(), total=len(df_answers), desc="Evaluating responses"):
        question = row.get("question", "")
        for col in answer_cols:
            ans = row.get(col, "")
            if not isinstance(ans, str) or not ans.strip():
                continue

            for metric in args.metrics:
                try:
                    res = judge.evaluate_metric(question, ans, metric)
                    results.append({
                        "question": question,
                        "answer": ans,
                        "answer_source": col,
                        "evaluator_model": evaluator_name,
                        "metric": metric,
                        "score": res.get("score"),
                        "reason": res.get("justification")
                    })
                except Exception as e:
                    print(f"Error evaluating {col} on {metric}: {e}", file=sys.stderr)

    out_df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Evaluation complete! Saved {len(out_df)} evaluations to {out_path}")


if __name__ == "__main__":
    main()
