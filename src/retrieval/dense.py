"""Dense semantic retrieval with SentenceTransformers and FAISS."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
try:
    import numpy as np
except ImportError:
    np = None

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    faiss = None
    SentenceTransformer = None
    torch = None


class DenseRetriever:
    """Dense retriever using SentenceTransformer embeddings and FAISS index."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        normalize_embeddings: bool = True,
        device: Optional[str] = None
    ):
        if SentenceTransformer is None or faiss is None:
            raise ImportError("sentence-transformers and faiss-cpu must be installed to use DenseRetriever.")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.normalize = normalize_embeddings
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.index: Optional[faiss.Index] = None
        self.is_e5 = "e5" in model_name.lower()

    def build_index(
        self,
        corpus_texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> faiss.Index:
        """Encodes corpus texts and constructs FAISS search index."""
        inputs = [("passage: " + t) if self.is_e5 else t for t in corpus_texts]
        embeddings = self.model.encode(
            inputs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim) if self.normalize else faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        return self.index

    def search(
        self,
        query: str,
        top_k: int = 30
    ) -> Tuple[List[int], List[float]]:
        """Searches index for nearest neighbors.

        Returns:
            Tuple of (hit_indices, hit_scores).
        """
        if self.index is None:
            raise ValueError("FAISS index has not been built or loaded.")

        q_text = ("query: " + query) if self.is_e5 else query
        q_emb = self.model.encode(
            [q_text],
            normalize_embeddings=self.normalize
        ).astype("float32")

        distances, indices = self.index.search(q_emb, top_k)
        return indices[0].tolist(), distances[0].tolist()

    def save_index(self, filepath: Union[str, Path]):
        """Saves FAISS index to disk."""
        if self.index is None:
            raise ValueError("No index to save.")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(filepath))

    def load_index(self, filepath: Union[str, Path]):
        """Loads FAISS index from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"FAISS index file not found: {filepath}")
        self.index = faiss.read_index(str(filepath))
