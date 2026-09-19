"""Kuwaiti Labor Law Benchmark (kw-labor-law-benchmark).

A reproducible research package for Arabic legal QA, Retrieval-Augmented Generation (RAG),
and LLM evaluation under Kuwaiti labor statutes.
"""

__version__ = "0.1.0"

from .config import BenchmarkConfig, load_config

__all__ = ["__version__", "BenchmarkConfig", "load_config"]
