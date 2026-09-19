"""Sparse lexical retrieval using BM25Okapi."""

from typing import List, Optional, Tuple
try:
    import numpy as np
except ImportError:
    np = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from data.preprocessor import simple_tokenize


class SparseBM25Retriever:
    """BM25 lexical retriever for Arabic legal texts."""

    def __init__(self, corpus_texts: Optional[List[str]] = None):
        if BM25Okapi is None:
            raise ImportError("rank-bm25 must be installed to use SparseBM25Retriever.")

        self.bm25: Optional[BM25Okapi] = None
        self.corpus_size = 0
        if corpus_texts is not None:
            self.build_index(corpus_texts)

    def build_index(self, corpus_texts: List[str]):
        """Tokenizes corpus and initializes BM25 index."""
        tokenized_corpus = [simple_tokenize(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_size = len(corpus_texts)

    def get_scores(self, query: str) -> np.ndarray:
        """Returns BM25 scores for all corpus documents."""
        if self.bm25 is None:
            raise ValueError("BM25 index has not been built.")
        q_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        return np.array(scores, dtype=np.float64)

    def search(self, query: str, top_k: int = 30) -> Tuple[List[int], List[float]]:
        """Returns top-k document indices and scores."""
        scores = self.get_scores(query)
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(-scores)[:top_k].tolist()
        top_scores = [float(scores[i]) for i in top_indices]
        return top_indices, top_scores
