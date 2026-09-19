# Data Directory Structure

This directory contains the datasets and corpora used by **SLED** (*Evaluating LLMs as Answerers and Judges for Low-Resource Domain-Specific Legal QAs*).

```
data/
├── samples/              # [TRACKED] Synthetic / sample data for testing and quick verification
│   ├── sample_cases.csv
│   └── sample_law_articles.csv
├── corpus/               # Extracted and structured statutory articles
└── generated/            # Model outputs and benchmark generations
```

## Ground Truth Data Policy
In accordance with research and privacy protocols, the full ground truth datasets (placed locally in `data/ground_truth/`) are not committed to the remote repository.

### Expected Schemas

#### 1. Legal Cases Gold (`legal_cases_gold.csv`):
- `#`: Case ID (int)
- `case_text`: The legal dispute or question text in Arabic.
- `Answer`: Verified expert legal opinion / reference court summary.
- `Source`: Case reference (e.g. Court of Cassation ruling number).
- `Articles`: Relevant statutory article numbers (e.g., `[24]`).

#### 2. Law Articles Corpus (`law_articles.csv`):
- `article_number`: Statutory article number (e.g., `1`, `24`, `55`).
- `article_text`: Full legal text of the statutory provision.
- Optional metadata: `law_name`, `title`, `url`, `source`.
