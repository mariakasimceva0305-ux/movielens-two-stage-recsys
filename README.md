# movielens-two-stage-recsys

Educational two-stage recommendation system on MovieLens data.

## Project Scope

Recommendation pipeline with candidate generation, ranking, evaluation, and API serving.

## Repository Structure

- `src/data.py`
- `src/candidate_generation.py`
- `src/features.py`
- `src/ranking.py`
- `src/train.py`
- `src/evaluate.py`
- `src/inference.py`
- `app/main.py`
- `Dockerfile`

## Implemented Functionality

- candidate generation
- ranking stage over generated candidates
- offline evaluation routines
- API entrypoint for recommendation requests