# MovieLens Two-Stage Recommender System

Production-minded educational implementation of a **two-stage recommendation pipeline** built on MovieLens data. The project separates candidate generation and ranking, evaluates recommendation quality with top-K metrics, and exposes inference through a FastAPI service.

## Why This Project
A single-stage recommender is often too limited when you need both **retrieval efficiency** and **top-K relevance**. This repository demonstrates a common industry pattern:
1. generate a compact candidate set fast;
2. apply a richer ranker on top;
3. validate improvements with offline metrics before serving recommendations.

## Problem Statement
Given user-item interaction history, generate a ranked list of relevant movies for a target user.

## Objectives
- implement a **two-stage recommender**
- keep training, feature engineering, inference, and evaluation modular
- compare recommendation quality using **offline ranking metrics**
- package the inference layer as a lightweight API

## Project Scope
This repository is intended as a compact reference implementation of:
- candidate generation
- ranking
- offline evaluation
- service packaging
- reproducible local execution

## Architecture

### Stage 1 — Candidate Generation
Generate a manageable candidate pool for each user using historical interactions and retrieval-style logic.

### Stage 2 — Ranking
Re-score candidate items with a ranking model using engineered user-item features.

### Serving Layer
Return ranked recommendations via FastAPI.

## Repository Structure
```text
app/
  main.py                 # FastAPI entrypoint
src/
  candidate_generation.py # candidate generation logic
  config.py               # configuration
  data.py                 # data loading / preparation
  evaluate.py             # offline metrics
  features.py             # feature engineering
  inference.py            # recommendation inference
  ranking.py              # ranking logic
  train.py                # training pipeline
artifacts/                # trained artifacts / intermediate outputs
notebooks/                # exploratory analysis
Dockerfile
requirements.txt
README.md
```

## ML Approach
### Candidate Generation
The first stage narrows the search space and keeps inference tractable.

### Ranking
The second stage reorders candidates using richer features than the retrieval layer can typically afford.

### Why Two-Stage
A two-stage setup is a practical trade-off between:
- **latency**
- **coverage**
- **quality of the final top-K list**

## Evaluation
The project tracks standard ranking metrics:
- `Precision@K`
- `Recall@K`
- `MAP@K`
- `NDCG@K`
- latency of the recommendation path

These metrics are useful because they separate:
- retrieval coverage,
- top-K ordering quality,
- and serving practicality.

## What This Repository Demonstrates
- modular recommender code instead of a single notebook-only workflow
- explicit split between training and inference
- ability to expose ranking results through an API
- engineering-friendly layout for future iteration

## Running Locally
```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
python -m src.train
uvicorn app.main:app --reload
```

## Example API Use
```bash
curl -X GET "http://127.0.0.1:8000/recommend?user_id=1&k=10"
```

## Suggested Improvements
- add explicit train / validation / test split documentation
- log metric values for baseline vs ranker in a dedicated results table
- add feature importance or error analysis summary
- support approximate nearest neighbor retrieval for larger candidate pools
- add experiment tracking for repeatable comparison runs

## Limitations
- MovieLens is a benchmark dataset, not a production catalog
- offline gains do not automatically translate to online business lift
- cold-start handling is outside the current scope
- candidate generation strategy can be extended further for scale

## Takeaway
This project is best read as a compact demonstration of how to structure a recommender system repo when the goal is not just to train a model, but to **evaluate it, package it, and make the system legible to another engineer**.
