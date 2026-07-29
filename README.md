# Zepto Data & AI Platform — Capstone Project

A single connected platform made of three modules: a data-engineering
pipeline, an end-to-end analytics/modeling pipeline, and a GenAI support
assistant — all graded independently but submitted together in this one
repository.

| Module | Path | Marks |
|---|---|---|
| Data Pipeline | [`/data_pipeline`](./data_pipeline) | 25 |
| Analytics Pipeline | [`/analytics`](./analytics) | 50 |
| Support Assistant | [`/support_assistant`](./support_assistant) | 25 |

## Setup

Each module has its own `requirements.txt`. Install per-module as you enter
each folder:

```bash
cd data_pipeline && pip install -r requirements.txt && cd ..
cd analytics && pip install -r requirements.txt && cd ..
cd support_assistant && pip install -r requirements.txt && cd ..
```

## How to run each module

### `/data_pipeline`
```bash
cd data_pipeline
python scrape.py            # -> raw_books.csv
python clean_and_load.py    # -> books.db
python queries.py           # -> SQL query results + pandas verification
```
Scrapes books.toscrape.com, cleans and converts prices to INR at a fixed
rate, loads into a normalized SQLite schema, and runs/verifies SQL queries.
See [`data_pipeline/README.md`](./data_pipeline/README.md) for full design
decisions.

### `/analytics`
```bash
cd analytics
python 01_eda.py         # loads Titanic data (once), profiles, cleans, EDA charts
python 02_modeling.py    # reads the cleaned data, full modeling pipeline
```
Profiles and cleans the Titanic dataset, tells a visual data story, then
trains/tunes/compares three classifiers plus a regression side-task, saving
the full fitted pipeline with `joblib`. See
[`analytics/README.md`](./analytics/README.md) for all written
interpretations, metrics, and the final model recommendation.

### `/support_assistant`
```bash
cd support_assistant
python ingest.py                              # sanity-checks embedding + retrieval
python -m uvicorn main:app --host 0.0.0.0 --port 7860
```
Then POST to `http://localhost:7860/ask` with `{"query": "..."}`. A RAG
assistant over Zepto's own policy documents, using sentence-transformers +
ChromaDB for retrieval and a LangGraph-orchestrated flow for intent
routing, served via FastAPI. Runs fully offline/deterministic by default
(`MOCK_LLM` unset) — no signup or API key needed. See
[`support_assistant/README.md`](./support_assistant/README.md) for the full
architecture description and example call transcripts. A `Dockerfile` is
included for local build-and-run.

## Design decisions summary

- **Data pipeline:** missing/malformed scraped fields are median-imputed
  rather than dropped, to avoid losing otherwise-valid rows; currency
  conversion uses a fixed, stated rate (1 GBP = 105.50 INR) rather than a
  live API, per the assignment's required baseline.
- **Analytics:** missing values are handled by an explicit percentage
  threshold rule (drop <5%, impute 5-30%, drop/flag column above that); all
  preprocessing for modeling is fit on the training split only, enforced via
  a scikit-learn `ColumnTransformer`/`Pipeline`, to avoid test-set leakage.
- **Support assistant:** every LLM call is gated behind a `MOCK_LLM` toggle,
  defaulting to a fully deterministic, keyless mock mode that is the graded
  baseline; retrieval (embeddings + ChromaDB) always runs for real in both
  modes since it needs no API key.

## Git workflow

Each module was developed on its own feature branch
(`feature/data-pipeline`, `feature/analytics`, `feature/support-assistant`),
committed to at least twice, then merged into `main` — visible via
`git log --graph --all`.
