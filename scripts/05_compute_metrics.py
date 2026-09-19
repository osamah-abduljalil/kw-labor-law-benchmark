#!/usr/bin/env python3
"""CLI Script 05: Aggregate evaluation metrics and calculate human correlation."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError:
    pd = None

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.metrics import (
    compute_human_correlation,
    compute_model_score_summary,
)


def run_with_csv(eval_path: str, human_path: str = None):
    """Fallback metric aggregator using standard library csv when pandas is absent."""
    with open(eval_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    # Group by metric and answer_source
    scores_by_group = defaultdict(list)
    for r in records:
        source = r.get("answer_source", "all")
        metric = r.get("metric", "unknown")
        raw_score = r.get("score")
        if raw_score:
            try:
                val = float(raw_score)
                scores_by_group[(source, metric)].append(val)
            except ValueError:
                pass

    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION SUMMARY (Python Standard Library)")
    print("=" * 60)
    header = f"{'Source':<30} | {'Metric':<15} | {'Count':<6} | {'Mean':<6} | {'Min':<5} | {'Max':<5}"
    print(header)
    print("-" * len(header))
    for (src, metric), s_list in sorted(scores_by_group.items()):
        if s_list:
            mean_val = round(sum(s_list) / len(s_list), 2)
            min_val = round(min(s_list), 1)
            max_val = round(max(s_list), 1)
            print(f"{src:<30} | {metric:<15} | {len(s_list):<6} | {mean_val:<6} | {min_val:<5} | {max_val:<5}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation scores and compare with human ratings.")
    parser.add_argument("--eval-results", "-e", type=str, required=True, help="Path to evaluation CSV (e.g. final_evaluation_results.csv)")
    parser.add_argument("--human-eval", type=str, default=None, help="Optional path to human scored CSV (e.g. all_human_scored_qa.csv)")
    parser.add_argument("--output-table", "-o", type=str, default=None, help="Optional path to save summary table CSV")

    args = parser.parse_args()

    if pd is None:
        run_with_csv(args.eval_results, args.human_eval)
        return

    eval_df = pd.read_csv(args.eval_results)
    summary = compute_model_score_summary(eval_df)

    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    if args.human_eval and Path(args.human_eval).exists():
        human_df = pd.read_csv(args.human_eval)
        corr = compute_human_correlation(eval_df, human_df)
        print("\n" + "=" * 60)
        print("CORRELATION WITH HUMAN EVALUATIONS")
        print("=" * 60)
        for m, stats in corr.items():
            print(f"Metric: {m.upper()}")
            print(f"  Sample size: {stats['n_samples']}")
            print(f"  MAE: {stats['mae']}")
            print(f"  Pearson Correlation: {stats['pearson']}")

    if args.output_table:
        out_p = Path(args.output_table)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_p, index=False)
        print(f"\nSummary table saved to: {out_p}")


if __name__ == "__main__":
    main()
