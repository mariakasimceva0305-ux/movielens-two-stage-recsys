from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

from src.candidate_generation import generate_candidate_frame
from src.features import build_ranking_dataset
from src.ranking import score_candidates


@dataclass
class InferenceBundle:
    popularity_model: object
    item_item_model: object
    ranker_artifacts: object
    seen_items_map: Dict[int, set]
    train_df: pd.DataFrame
    movies_df: pd.DataFrame


def load_bundle(artifacts_dir: Path) -> InferenceBundle:
    popularity_model = joblib.load(artifacts_dir / "popularity_model.joblib")
    item_item_model = joblib.load(artifacts_dir / "item_item_model.joblib")
    ranker_artifacts = joblib.load(artifacts_dir / "ranker_artifacts.joblib")
    seen_items_map = joblib.load(artifacts_dir / "seen_items_map.joblib")
    train_df = joblib.load(artifacts_dir / "train_df.joblib")
    movies_df = joblib.load(artifacts_dir / "movies_df.joblib")

    return InferenceBundle(
        popularity_model=popularity_model,
        item_item_model=item_item_model,
        ranker_artifacts=ranker_artifacts,
        seen_items_map=seen_items_map,
        train_df=train_df,
        movies_df=movies_df,
    )


def recommend_for_user(bundle: InferenceBundle, user_id: int, k: int = 10) -> List[Dict]:
    seen_items = bundle.seen_items_map.get(user_id, set())

    # Cold start fallback.
    if user_id not in bundle.seen_items_map:
        movie_lookup = bundle.movies_df.set_index("movieId")
        return [
            {
                "movie_id": movie_id,
                "title": movie_lookup.loc[movie_id, "title"] if movie_id in movie_lookup.index else str(movie_id),
                "score": None,
            }
            for movie_id in bundle.popularity_model.recommend(seen_items=set(), k=k)
        ]

    candidates = generate_candidate_frame(
        users=[user_id],
        seen_items_map=bundle.seen_items_map,
        popularity_model=bundle.popularity_model,
        item_item_model=bundle.item_item_model,
        top_k_candidates=max(100, k),
    )
    if candidates.empty:
        movie_lookup = bundle.movies_df.set_index("movieId")
        return [
            {
                "movie_id": int(movie_id),
                "title": movie_lookup.loc[movie_id, "title"] if movie_id in movie_lookup.index else str(movie_id),
                "score": None,
            }
            for movie_id in bundle.popularity_model.recommend(seen_items=seen_items, k=k)
        ]

    ranking_dataset = build_ranking_dataset(
        candidates_df=candidates,
        train_df=bundle.train_df,
        target_df=pd.DataFrame(columns=["userId", "movieId"]),
        movies=bundle.movies_df,
    )
    if ranking_dataset.empty:
        movie_lookup = bundle.movies_df.set_index("movieId")
        return [
            {
                "movie_id": int(movie_id),
                "title": movie_lookup.loc[movie_id, "title"] if movie_id in movie_lookup.index else str(movie_id),
                "score": None,
            }
            for movie_id in bundle.popularity_model.recommend(seen_items=seen_items, k=k)
        ]

    scored = score_candidates(ranking_dataset, bundle.ranker_artifacts).head(k)
    merged = scored.merge(bundle.movies_df[["movieId", "title"]], on="movieId", how="left", suffixes=("", "_meta"))

    results = []
    for _, row in merged.iterrows():
        results.append(
            {
                "movie_id": int(row["movieId"]),
                "title": row.get("title_meta") or row.get("title") or str(row["movieId"]),
                "score": float(row["rank_score"]),
            }
        )
    return results
