# movielens-two-stage-recsys

Educational two-stage recommendation system on MovieLens data.

## Repository Contents

- `src/data.py` - data loading and preprocessing.
- `src/candidate_generation.py` - candidate generation logic.
- `src/features.py` - feature engineering.
- `src/ranking.py` - ranking model logic.
- `src/train.py` - training pipeline.
- `src/evaluate.py` - offline evaluation routines.
- `src/inference.py` - recommendation inference flow.
- `app/main.py` - API entrypoint.
- `Dockerfile` - containerization setup.

## Implemented Functionality

The code implements:

- candidate generation for recommendation,
- ranking-based reordering of candidates,
- offline metric evaluation,
- API-level serving entrypoint for recommendation requests.
