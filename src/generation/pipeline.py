"""RAG generation pipeline over question sets."""

import json
from typing import Any, Dict, List, Optional
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from data.corpus import LegalCorpus
from models.base import BaseLLM
from retrieval.hybrid import HybridRetriever
from .prompt_builder import (
    build_legal_analyst_json_prompt,
    build_non_rag_prompt_ar,
    build_rag_prompt_ar,
)


class RAGPipeline:
    """Executes end-to-end RAG and Non-RAG generation benchmarks."""

    def __init__(
        self,
        retriever: HybridRetriever,
        corpus: LegalCorpus,
        models: Dict[str, BaseLLM],
        force_citations: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens: int = 500,
        structured_json: bool = False
    ):
        self.retriever = retriever
        self.corpus = corpus
        self.models = models
        self.force_citations = force_citations
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.structured_json = structured_json

    def answer_question(
        self,
        question: str,
        top_k_final: int = 8
    ) -> Dict[str, Any]:
        """Retrieves context and generates answers across all configured models."""
        evidence_ids, debug = self.retriever.retrieve(question, top_k_final=top_k_final)
        evidences = [self.corpus.get_document(i) for i in evidence_ids if i < len(self.corpus)]

        if self.structured_json:
            rag_prompt = build_legal_analyst_json_prompt(question, evidences)
        else:
            rag_prompt = build_rag_prompt_ar(question, evidences, force_citations=self.force_citations)

        answers = {}
        for model_name, model in self.models.items():
            ans = model.generate(
                rag_prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens
            )
            answers[model_name] = ans

        return {
            "evidence_ids": evidence_ids,
            "evidence_ids_with_scores": debug.get("evidence_ids_with_scores", []),
            "answers": answers,
            "rag_prompt": rag_prompt
        }

    def run_benchmark(
        self,
        questions_df: pd.DataFrame,
        q_col: str = "case_text",
        top_k_final: int = 8,
        include_non_rag: bool = True
    ) -> pd.DataFrame:
        """Runs generation across all questions in the dataset."""
        rows = []

        for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Running benchmark"):
            q = row[q_col]
            if not isinstance(q, str) or not q.strip():
                continue

            # RAG Generation
            rag_res = self.answer_question(q, top_k_final=top_k_final)
            row_dict: Dict[str, Any] = {
                "question": q,
                "evidence_ids": json.dumps(rag_res["evidence_ids"], ensure_ascii=False),
                "evidence_ids_with_scores": json.dumps(rag_res.get("evidence_ids_with_scores", []), ensure_ascii=False),
            }

            # Add RAG answers (matches format in rag_answers_all_evd_score.csv)
            for model_name, ans in rag_res["answers"].items():
                row_dict[f"answer_{model_name}"] = ans

            # Non-RAG Generation
            if include_non_rag:
                non_rag_prompt = build_non_rag_prompt_ar(q)
                for model_name, model in self.models.items():
                    ans_non_rag = model.generate(
                        non_rag_prompt,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_new_tokens=self.max_new_tokens
                    )
                    row_dict[f"answer_{model_name}_non_rag"] = ans_non_rag

            rows.append(row_dict)

        return pd.DataFrame(rows)
