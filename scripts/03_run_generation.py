#!/usr/bin/env python3
"""CLI Script 03: Run RAG and Non-RAG generation (batch benchmark or single query)."""

import argparse
from pathlib import Path
import sys

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import pandas as pd
except ImportError:
    pd = None

from config import load_config
from data.corpus import LegalCorpus
from data.preprocessor import clean_question
from generation.pipeline import RAGPipeline
from models.factory import load_models_from_config
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import SparseBM25Retriever


def main():
    parser = argparse.ArgumentParser(description="Run RAG and baseline non-RAG generation (single query or batch CSV).")
    parser.add_argument("--config", "-c", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--query", type=str, default=None, help="Run generation on a single query string instead of CSV")
    parser.add_argument("--questions", "-q", type=str, default=None, help="Questions CSV path (overrides config)")
    parser.add_argument("--corpus", type=str, default=None, help="Corpus CSV path (overrides config)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV path for batch generation")
    parser.add_argument("--top-k", type=int, default=None, help="Final retrieved chunks (top_k_final)")
    parser.add_argument("--no-non-rag", action="store_true", help="Skip non-RAG baseline generation")
    parser.add_argument("--structured-json", action="store_true", help="Format RAG prompt to output structured JSON")

    args = parser.parse_args()
    cfg = load_config(args.config)

    # Resolve corpus path
    corpus_candidates = [
        args.corpus,
        cfg.paths.law_csv_path,
        "data/corpus/law_articles.csv",
        "data/samples/sample_law_articles.csv",
    ]
    corpus_path = None
    for c in corpus_candidates:
        if c and Path(c).exists():
            corpus_path = c
            break

    if not corpus_path:
        print("Error: Could not locate law articles corpus. Please specify --corpus.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading legal corpus from: {corpus_path}")
    corpus = LegalCorpus.from_csv(
        csv_path=corpus_path,
        text_col=cfg.data.law_text_col,
        id_col=cfg.data.law_id_col,
        do_chunk=cfg.data.chunking.enabled
    )
    print(f"Loaded {len(corpus)} corpus documents.")

    # Initialize retrievers
    print("Initializing Dense retriever...")
    dense = DenseRetriever(
        model_name=cfg.retrieval.emb_name,
        normalize_embeddings=cfg.retrieval.normalize_emb
    )
    index_file = Path(cfg.paths.index_dir) / "faiss_index.bin"
    if index_file.exists():
        print(f"Loading cached FAISS index from {index_file}...")
        dense.load_index(index_file)
    else:
        print("Building in-memory FAISS index...")
        dense.build_index(corpus.texts)

    sparse = None
    if cfg.retrieval.use_bm25:
        print("Initializing BM25 retriever...")
        sparse = SparseBM25Retriever(corpus.texts)

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        corpus_texts=corpus.texts,
        alpha_dense=cfg.retrieval.alpha_dense
    )

    # Initialize models
    print("Initializing LLM models from config...")
    models = load_models_from_config(cfg.models)
    if not models:
        print("Error: No models could be loaded. Please check API keys in .env or config.", file=sys.stderr)
        sys.exit(1)
    print(f"Active models: {list(models.keys())}")

    top_k = args.top_k or cfg.retrieval.top_k_final

    pipeline = RAGPipeline(
        retriever=hybrid,
        corpus=corpus,
        models=models,
        force_citations=cfg.generation.force_citations,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        max_new_tokens=cfg.generation.max_new_tokens,
        structured_json=args.structured_json
    )

    # --- Mode 1: Single interactive query ---
    if args.query:
        query_text = clean_question(args.query)
        print(f"\nEvaluating single query: '{query_text}'")
        res = pipeline.answer_question(query_text, top_k_final=top_k)

        print("\n" + "=" * 60)
        print(f"RETRIEVED EVIDENCE (Top {len(res['evidence_ids'])})")
        print("=" * 60)
        for rank, doc_id in enumerate(res["evidence_ids"], 1):
            text, meta = corpus.get_document(doc_id)
            print(f"[{rank}] المادة {meta.get('article', 'N/A')}: {text[:140]}...")

        print("\n" + "=" * 60)
        print("GENERATED MODEL ANSWERS")
        print("=" * 60)
        for model_name, answer in res["answers"].items():
            print(f"\n--- Model: {model_name} ---")
            print(answer)
        return

    # --- Mode 2: Batch CSV generation ---
    question_candidates = [
        args.questions,
        cfg.paths.qa_csv_path,
        "data/ground_truth/legal_cases_gold.csv",
        "data/samples/sample_cases.csv",
    ]
    questions_path = None
    for q in question_candidates:
        if q and Path(q).exists():
            questions_path = q
            break

    if not questions_path:
        print("Error: Could not locate questions dataset. Please specify --questions.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output or (Path(cfg.paths.output_dir) / "rag_benchmark_answers.csv"))

    if pd is None:
        print("Error: pandas is required for batch CSV generation. Please install pandas.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading questions from: {questions_path}")
    questions_df = pd.read_csv(questions_path)
    q_col = cfg.data.q_col
    if q_col not in questions_df.columns:
        # Fallback to 'Question' if case_text is not present
        if "Question" in questions_df.columns:
            q_col = "Question"
        else:
            raise ValueError(f"Questions CSV must contain '{q_col}' or 'Question'. Found: {questions_df.columns.tolist()}")

    # Clean question column (strip 'س 1 / ...' prefixes like notebook Cell 6)
    questions_df[q_col] = questions_df[q_col].astype(str).apply(clean_question)
    print(f"Loaded and cleaned {len(questions_df)} questions.")

    results_df = pipeline.run_benchmark(
        questions_df=questions_df,
        q_col=q_col,
        top_k_final=top_k,
        include_non_rag=not args.no_non_rag
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Generation complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
