"""Evaluation modules and metric computations."""

from .judge import LegalBenchmarkJudge
from .metrics import (
    compute_human_correlation,
    compute_model_score_summary,
    format_benchmark_table_markdown,
)

__all__ = [
    "LegalBenchmarkJudge",
    "compute_model_score_summary",
    "compute_human_correlation",
    "format_benchmark_table_markdown",
]
