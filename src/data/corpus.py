"""Corpus loading and index metadata management for Kuwaiti Labor Law."""

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pandas as pd
except ImportError:
    pd = None

from .preprocessor import chunk_text_words, normalize_arabic


class LegalCorpus:
    """Manages legal text corpus, chunking, and associated document metadata."""

    def __init__(
        self,
        corpus_texts: List[str],
        corpus_meta: List[Dict[str, Any]]
    ):
        self.texts = corpus_texts
        self.metadata = corpus_meta

    def __len__(self) -> int:
        return len(self.texts)

    def get_document(self, index: int) -> Tuple[str, Dict[str, Any]]:
        return self.texts[index], self.metadata[index]

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        text_col: str = "article_text",
        id_col: str = "article_number",
        do_chunk: bool = False,
        chunk_words: int = 220,
        overlap_words: int = 40,
        law_name: str = "قانون العمل في القطاع الأهلي رقم 6 لسنة 2010"
    ) -> "LegalCorpus":
        """Loads corpus from a CSV file.

        Args:
            csv_path: Path to articles CSV.
            text_col: Name of column containing article text.
            id_col: Name of column containing article number/ID.
            do_chunk: Whether to further sub-chunk each article.
            chunk_words: Max words per chunk when chunking.
            overlap_words: Overlap words per chunk.
            law_name: Default law title for evidence header metadata.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Corpus CSV not found at: {csv_path}")

        corpus_texts: List[str] = []
        corpus_meta: List[Dict[str, Any]] = []

        if pd is not None:
            df = pd.read_csv(csv_path)
            if text_col not in df.columns:
                raise ValueError(f"Corpus CSV must contain column '{text_col}'. Found: {df.columns.tolist()}")

            base_meta_cols = [c for c in ["doc_id", "title", "law_name", "article", "article_number", "url", "source"] if c in df.columns]

            for idx, row in df.iterrows():
                text = row[text_col]
                if not isinstance(text, str) or not text.strip():
                    continue

                meta = {c: row[c] for c in base_meta_cols}
                meta["row_index"] = int(idx)
                if "article" not in meta and id_col in df.columns:
                    meta["article"] = str(row[id_col])
                if "law_name" not in meta or pd.isna(meta.get("law_name")):
                    meta["law_name"] = law_name

                if do_chunk:
                    chunks = chunk_text_words(text, chunk_words=chunk_words, overlap_words=overlap_words)
                    for j, ch in enumerate(chunks):
                        corpus_texts.append(ch)
                        m = dict(meta)
                        m["chunk_id"] = j
                        corpus_meta.append(m)
                else:
                    corpus_texts.append(normalize_arabic(text))
                    meta["chunk_id"] = 0
                    corpus_meta.append(meta)
        else:
            # Fallback to standard library csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                if text_col not in fieldnames:
                    raise ValueError(f"Corpus CSV must contain column '{text_col}'. Found: {fieldnames}")

                base_meta_cols = [c for c in ["doc_id", "title", "law_name", "article", "article_number", "url", "source"] if c in fieldnames]

                for idx, row in enumerate(reader):
                    text = row.get(text_col, "")
                    if not text or not text.strip():
                        continue

                    meta = {c: row.get(c) for c in base_meta_cols}
                    meta["row_index"] = int(idx)
                    if "article" not in meta and id_col in row:
                        meta["article"] = str(row.get(id_col, ""))
                    if not meta.get("law_name"):
                        meta["law_name"] = law_name

                    if do_chunk:
                        chunks = chunk_text_words(text, chunk_words=chunk_words, overlap_words=overlap_words)
                        for j, ch in enumerate(chunks):
                            corpus_texts.append(ch)
                            m = dict(meta)
                            m["chunk_id"] = j
                            corpus_meta.append(m)
                    else:
                        corpus_texts.append(normalize_arabic(text))
                        meta["chunk_id"] = 0
                        corpus_meta.append(meta)

        return cls(corpus_texts, corpus_meta)
