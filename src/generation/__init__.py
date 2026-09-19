"""Generation modules and pipelines."""

from .pipeline import RAGPipeline
from .prompt_builder import (
    build_legal_analyst_json_prompt,
    build_non_rag_prompt_ar,
    build_rag_prompt_ar,
    format_evidence_block,
)

__all__ = [
    "format_evidence_block",
    "build_rag_prompt_ar",
    "build_non_rag_prompt_ar",
    "build_legal_analyst_json_prompt",
    "RAGPipeline",
]
