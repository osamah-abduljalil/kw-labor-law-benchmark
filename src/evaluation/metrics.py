from typing import Dict, Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None


def compute_model_score_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates judge scores by evaluator model, answer source, and metric.

    Expected columns in eval_df:
        - answer_source (model name)
        - metric (completeness, correctness, fluency)
        - score (numerical 1-5)
    """
    if "score" not in eval_df.columns or "metric" not in eval_df.columns:
        raise ValueError("DataFrame must contain 'score' and 'metric' columns.")

    group_cols = ["metric"]
    if "answer_source" in eval_df.columns:
        group_cols.insert(0, "answer_source")

    summary = (
        eval_df.groupby(group_cols)["score"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    summary["mean"] = summary["mean"].round(2)
    summary["std"] = summary["std"].round(2)
    return summary


def compute_human_correlation(
    llm_eval_df: pd.DataFrame,
    human_eval_df: pd.DataFrame,
    join_col: str = "question"
) -> Dict[str, Dict[str, float]]:
    """Calculates MAE and correlation between automated LLM judge and human scores.

    Args:
        llm_eval_df: DataFrame of LLM judge results.
        human_eval_df: DataFrame of human scored answers.
        join_col: Column to merge evaluations on.

    Returns:
        Dict mapping each metric to correlation statistics (MAE, Pearson, Spearman).
    """
    results = {}
    metrics = ["completeness", "correctness", "fluency"]

    # Normalize column names in human evaluation df
    human_df = human_eval_df.copy()
    human_df.columns = [c.strip().lower() for c in human_df.columns]

    for m in metrics:
        if m not in human_df.columns:
            continue

        # Filter llm_df for this metric
        sub_llm = llm_eval_df[llm_eval_df["metric"].str.lower() == m]
        merged = pd.merge(
            sub_llm,
            human_df,
            on=join_col,
            suffixes=("_llm", "_human")
        )

        if len(merged) < 2:
            continue

        s_llm = pd.to_numeric(merged["score_llm"], errors="coerce")
        s_human = pd.to_numeric(merged[m], errors="coerce")
        valid = s_llm.notna() & s_human.notna()

        y_llm = s_llm[valid]
        y_human = s_human[valid]

        mae = float(np.mean(np.abs(y_llm - y_human)))
        pearson = float(np.corrcoef(y_llm, y_human)[0, 1]) if len(y_llm) > 1 else np.nan

        results[m] = {
            "n_samples": int(valid.sum()),
            "mae": round(mae, 3),
            "pearson": round(pearson, 3)
        }

    return results


def format_benchmark_table_markdown(summary_df: pd.DataFrame) -> str:
    """Formats summary dataframe as a clean markdown table."""
    return summary_df.to_markdown(index=False)
