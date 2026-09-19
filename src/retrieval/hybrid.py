"""Hybrid dense + sparse retrieval combining FAISS and BM25."""

from typing import Any, Dict, List, Optional, Tuple
try:
    import numpy as np
except ImportError:
    np = None
from data.preprocessor import normalize_arabic
from .dense import DenseRetriever
from .sparse import SparseBM25Retriever


class HybridRetriever:
    """Combines dense semantic embeddings and BM25 lexical matching."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: Optional[SparseBM25Retriever] = None,
        corpus_texts: Optional[List[str]] = None,
        alpha_dense: float = 0.65
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.corpus_texts = corpus_texts or []
        self.alpha_dense = alpha_dense

    def retrieve(
        self,
        question: str,
        top_k_dense: int = 30,
        top_k_final: int = 8,
        alpha_dense: Optional[float] = None
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Performs hybrid retrieval for a question.

        Args:
            question: Legal question string.
            top_k_dense: Number of candidate documents to retrieve densely.
            top_k_final: Number of final document IDs to return.
            alpha_dense: Weight for dense score (0..1). If None, uses default.

        Returns:
            Tuple of (top_document_indices, debug_info_dict).
        """
        if alpha_dense is None:
            alpha_dense = self.alpha_dense

        q_norm = normalize_arabic(question)

        # Dense retrieval
        dense_ids, dense_scores = self.dense.search(q_norm, top_k=top_k_dense)

        # If sparse retriever is not provided, fall back to pure dense
        if self.sparse is None:
            final_ids = dense_ids[:top_k_final]
            return final_ids, {
                "dense_ids": dense_ids,
                "dense_scores": dense_scores,
                "alpha_dense": 1.0
            }

        # BM25 scores across corpus
        bm25_scores = self.sparse.get_scores(q_norm)
        num_docs = len(self.corpus_texts) if self.corpus_texts else len(bm25_scores)

        # Normalize dense scores among candidate set
        dense_arr = np.full((num_docs,), -1e9, dtype=np.float64)
        for doc_id, sc in zip(dense_ids, dense_scores):
            if doc_id < num_docs:
                dense_arr[doc_id] = sc

        cand_dense = np.array([dense_arr[i] for i in dense_ids if i < num_docs], dtype=np.float64)
        if cand_dense.size > 0:
            d_min, d_max = cand_dense.min(), cand_dense.max()
            if d_max > d_min:
                for doc_id in dense_ids:
                    if doc_id < num_docs:
                        dense_arr[doc_id] = (dense_arr[doc_id] - d_min) / (d_max - d_min)
            else:
                for doc_id in dense_ids:
                    if doc_id < num_docs:
                        dense_arr[doc_id] = 1.0

        # Normalize BM25 scores globally
        b_min, b_max = bm25_scores.min(), bm25_scores.max()
        if b_max > b_min:
            bm25_norm = (bm25_scores - b_min) / (b_max - b_min)
        else:
            bm25_norm = np.zeros_like(bm25_scores)

        # Union of dense candidates + top BM25 candidates
        top_bm25_ids = np.argsort(-bm25_norm)[:top_k_dense].tolist()
        union_ids = list(set([i for i in (dense_ids + top_bm25_ids) if i < num_docs]))

        # Calculate hybrid scores
        hybrid_scores = []
        for doc_id in union_ids:
            score = alpha_dense * dense_arr[doc_id] + (1.0 - alpha_dense) * bm25_norm[doc_id]
            hybrid_scores.append(score)

        # Sort descending
        sorted_pairs = sorted(zip(union_ids, hybrid_scores), key=lambda x: x[1], reverse=True)
        final_ids = [doc_id for doc_id, _ in sorted_pairs[:top_k_final]]
        final_ids_with_scores = [(doc_id, round(float(score), 4)) for doc_id, score in sorted_pairs[:top_k_final]]

        debug_info = {
            "dense_ids": dense_ids,
            "dense_scores": dense_scores,
            "top_bm25_ids": top_bm25_ids[:10],
            "alpha_dense": alpha_dense,
            "evidence_ids_with_scores": final_ids_with_scores
        }

        return final_ids, debug_info
