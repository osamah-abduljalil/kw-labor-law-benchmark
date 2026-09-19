# SLED: Evaluating LLMs as Answerers and Judges for Low-Resource Domain-Specific Legal QAs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Domain: Legal NLP](https://img.shields.io/badge/Domain-Arabic%20Legal%20NLP-green.svg)](#)

A reproducible research benchmark and framework for evaluating **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)** on Arabic legal reasoning under the **Kuwaiti Labor Law (Law No. 6 of 2010)**.

---

## Overview

Legal QA in civil law jurisdictions demands precise grounding in statutory provisions and court precedents. **SLED** provides a comprehensive framework and benchmark to evaluate:
1. **Hybrid Retrieval**: Dense multilingual embeddings (`intfloat/multilingual-e5-large` with FAISS) combined with sparse lexical matching (Okapi BM25) for Arabic statutory law.
2. **LLMs as Answerers**: Comparing state-of-the-art LLMs (GPT-4o, DeepSeek-Chat, LLaMA-3.1-70B, Qwen-2.5-72B, ALLaM-7B, Gemini-2.5-Flash) across RAG-grounded and parametric (non-RAG) settings under the **Kuwaiti Labor Law (Law No. 6 of 2010)**.
3. **LLMs as Judges**: Automated multi-dimensional legal evaluation against expert legal rubrics:
   - **Completeness**: Coverage of all necessary legal elements and statutory issues.
   - **Correctness**: Statutory interpretation and alignment with legal provisions.
   - **Fluency & Legal Drafting**: Professional Arabic legal discourse and structure.


---

## Repository Structure

```
kw-labor-law-benchmark/
├── .gitignore                     # Ignores private ground truth datasets, .env, caches
├── .env.example                   # Template for API keys (DeepSeek, Gemini, HF, OpenAI)
├── pyproject.toml                 # Package definition (pip install -e .)
├── requirements.txt               # Pinned dependencies
├── config/
│   └── default_config.yaml        # Central configuration (retrieval, model, generation params)
├── data/
│   ├── README.md                  # Data layout and schema documentation
│   ├── samples/                   # [TRACKED] Mock/sample data for testing & reproduction
│   │   ├── sample_cases.csv
│   │   └── sample_law_articles.csv
│   ├── corpus/                    # Parsed statutory articles
│   └── generated/                 # Generated benchmark outputs
│       └── rag_answers_all_evd_score.csv
├── evaluations/                   # Evaluation results
│   ├── generated evaluations/     # LLM-as-a-judge multi-round outputs
│   └── human evaluations/         # Human expert scored QA
├── prompts/                       # Modular prompt templates
│   ├── generation/                # Legal analysis prompt
│   │   └── prompt.md
│   └── judge/                     # LLM-as-a-judge rubrics
│       ├── completeness.md
│       ├── correctness.md
│       └── fluency.md
├── src/                           # Core research modules
│   ├── config.py                  # Configuration & environment loader
│   ├── data/                      # Preprocessor, extractor, corpus management
│   ├── retrieval/                 # Dense (FAISS), Sparse (BM25), Hybrid retrievers
│   ├── models/                    # Unified LLM interfaces (HF, DeepSeek, Gemini, OpenAI)
│   ├── generation/                # Prompt builders & RAG pipeline
│   └── evaluation/                # LLM-as-a-judge & metric aggregation
├── scripts/                       # Reproducible CLI pipeline
│   ├── 01_extract_articles.py     # Extract articles from raw legal text
│   ├── 02_build_index.py          # Pre-encode and save FAISS index
│   ├── 03_run_generation.py       # Run RAG & Non-RAG answer generation (batch or single query)
│   ├── 04_run_evaluation.py       # Run LLM-as-a-judge automated scoring (batch or single Q&A)
│   └── 05_compute_metrics.py      # Aggregate scores and compute human correlation
```

---

## Installation & Setup

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/osamah-abduljalil/kw-labor-law-benchmark.git
cd kw-labor-law-benchmark

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. Install Package

Install dependencies and the `kw-labor-law-benchmark` package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Configure API Credentials

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
DEEPSEEK_API_KEY=your_deepseek_api_key
GOOGLE_API_KEY=your_gemini_api_key
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```

---

## Quickstart: Reproducing the Pipeline via CLI

The research benchmark pipeline can be run from start to finish via modular CLI commands:

### Step 1: Extract Legal Articles (Optional)
Extract articles from raw text (`law_text.txt`):
```bash
python scripts/01_extract_articles.py \
  --input raw_law_text.txt \
  --output data/corpus/law_articles.csv
```

### Step 2: Build & Cache Dense FAISS Index
Encode the corpus using multilingual E5 and persist the FAISS index:
```bash
python scripts/02_build_index.py \
  --config config/default_config.yaml \
  --corpus data/samples/sample_law_articles.csv \
  --output-dir data/index/
```

### Step 3: Run RAG & Non-RAG Generation
Generate answers across configured LLMs:
```bash
# Batch generation over questions CSV:
python scripts/03_run_generation.py \
  --config config/default_config.yaml \
  --questions data/ground_truth/legal_cases_gold.csv \
  --output data/generated/benchmark_answers.csv

# Or test a single query interactively:
python scripts/03_run_generation.py \
  --query "ما هي إجازة الوضع للمرأة الحامل وفقاً لقانون العمل؟"
```

### Step 4: Run Automated LLM-as-a-Judge Evaluation
Evaluate model answers on Completeness, Correctness, and Fluency:
```bash
# Batch evaluation over generated answers CSV:
python scripts/04_run_evaluation.py \
  --answers data/generated/benchmark_answers.csv \
  --evaluator deepseek-chat \
  --output evaluations/evaluation_results.csv

# Or evaluate a single Q&A directly:
python scripts/04_run_evaluation.py \
  --question "ما هي إجازة الوضع للمرأة الحامل؟" \
  --answer "تستحق المرأة العاملة إجازة 70 يوماً مدفوعة الأجر وفقاً للمادة 24." \
  --metrics completeness correctness fluency
```

### Step 5: Aggregate Metrics & Compare with Human Evaluations
Compute benchmark summary statistics and correlation with human scores:
```bash
python scripts/05_compute_metrics.py \
  --eval-results "evaluations/generated evaluations/3rd Round/final_evaluation_results.csv" \
  --human-eval "evaluations/human evaluations/all_human_scored_qa.csv" \
  --output-table "evaluations/benchmark_summary.csv"
```

---

## Python API Usage

The research modules in `src/` can be imported directly into Python scripts and custom research workflows:

```python
from config import load_config
from data.corpus import LegalCorpus
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseBM25Retriever
from retrieval.hybrid import HybridRetriever
from models.factory import load_models_from_config
from generation.pipeline import RAGPipeline

# 1. Load configuration and legal corpus
cfg = load_config("config/default_config.yaml")
corpus = LegalCorpus.from_csv(cfg.paths.law_csv_path)

# 2. Setup hybrid retrieval
dense = DenseRetriever(cfg.retrieval.emb_name)
dense.build_index(corpus.texts)
sparse = SparseBM25Retriever(corpus.texts)
retriever = HybridRetriever(dense, sparse, corpus.texts, alpha_dense=0.65)

# 3. Load LLMs and initialize pipeline
models = load_models_from_config(cfg.models)
pipeline = RAGPipeline(retriever, corpus, models)

# 4. Answer a legal query
query = "ما هي إجازة الوضع للمرأة الحامل وفقاً لقانون العمل الكويتي؟"
result = pipeline.answer_question(query)
print(result["answers"]["deepseek_api"])
```

---

## Data Policy & Ground Truth Access

- **Ground Truth Protection**: Full gold standard legal cases (`legal_cases_gold.csv` and `Dataset Part B.csv`) are ignored by Git to adhere to licensing and research privacy requirements.
- **Samples Included**: Synthetic sample cases (`data/samples/sample_cases.csv`) and sample law articles (`data/samples/sample_law_articles.csv`) are provided in the repository for immediate testing and verification.
- **Using Custom Datasets**: Place your datasets in `data/ground_truth/` and update `config/default_config.yaml` or pass the `--questions` argument to the CLI scripts.

---

## Citation

If you use this benchmark, code, or findings in your research, please cite our paper:

```bibtex
@article{abduljalil2026sled,
  title={SLED: Evaluating LLMs as Answerers and Judges for Low-Resource Domain-Specific Legal QAs},
  author={Abduljalil, Osamah and Alfurih, Hessah and Alhoshan, Waad},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/osamah-abduljalil/kw-labor-law-benchmark}}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
