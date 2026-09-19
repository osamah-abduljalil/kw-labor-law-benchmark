#!/usr/bin/env python3
"""CLI Script 02: Build FAISS dense index for corpus."""

import argparse
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from data.corpus import LegalCorpus
from retrieval.dense import DenseRetriever


def main():
    parser = argparse.ArgumentParser(description="Build and persist FAISS dense search index.")
    parser.add_argument("--config", "-c", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--corpus", type=str, default=None, help="Path to law articles CSV (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save FAISS index (overrides config)")
    parser.add_argument("--model-name", type=str, default=None, help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")

    args = parser.parse_args()
    cfg = load_config(args.config)

    corpus_path = args.corpus or cfg.paths.law_csv_path
    output_dir = Path(args.output_dir or cfg.paths.index_dir)
    emb_model = args.model_name or cfg.retrieval.emb_name

    print(f"Loading corpus from: {corpus_path}")
    corpus = LegalCorpus.from_csv(
        csv_path=corpus_path,
        text_col=cfg.data.law_text_col,
        id_col=cfg.data.law_id_col,
        do_chunk=cfg.data.chunking.enabled,
        chunk_words=cfg.data.chunking.chunk_words,
        overlap_words=cfg.data.chunking.overlap_words
    )
    print(f"Loaded {len(corpus)} corpus chunks.")

    print(f"Building dense index using: {emb_model}")
    dense_retriever = DenseRetriever(
        model_name=emb_model,
        normalize_embeddings=cfg.retrieval.normalize_emb
    )
    dense_retriever.build_index(corpus.texts, batch_size=args.batch_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / "faiss_index.bin"
    dense_retriever.save_index(index_file)
    print(f"FAISS index saved successfully to: {index_file}")


if __name__ == "__main__":
    main()
