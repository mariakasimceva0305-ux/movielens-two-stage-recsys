from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import DataConfig, SplitConfig


REQUIRED_RATINGS_COLUMNS = {"userId", "movieId", "rating", "timestamp"}
REQUIRED_MOVIES_COLUMNS = {"movieId", "title", "genres"}


def load_movielens(data_config: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens ratings and movies tables."""
    ratings_path = Path(data_config.data_dir) / data_config.ratings_filename
    movies_path = Path(data_config.data_dir) / data_config.movies_filename

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    missing_ratings = REQUIRED_RATINGS_COLUMNS - set(ratings.columns)
    missing_movies = REQUIRED_MOVIES_COLUMNS - set(movies.columns)

    if missing_ratings:
        raise ValueError(f"ratings.csv missing columns: {missing_ratings}")
    if missing_movies:
        raise ValueError(f"movies.csv missing columns: {missing_movies}")

    return ratings, movies


def convert_to_implicit(ratings: pd.DataFrame, min_rating: float = 4.0) -> pd.DataFrame:
    """Convert explicit ratings into implicit binary feedback.

    Strategy:
    - keep interactions with rating >= min_rating
    - assign implicit label = 1
    """
    implicit_df = ratings.loc[ratings["rating"] >= min_rating, ["userId", "movieId", "timestamp"]].copy()
    implicit_df["target"] = 1
    implicit_df["datetime"] = pd.to_datetime(implicit_df["timestamp"], unit="s")
    implicit_df = implicit_df.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    return implicit_df


def filter_active_users(implicit_df: pd.DataFrame, min_user_interactions: int = 3) -> pd.DataFrame:
    counts = implicit_df.groupby("userId")["movieId"].size()
    active_users = counts[counts >= min_user_interactions].index
    return implicit_df[implicit_df["userId"].isin(active_users)].copy()


def time_based_split(interactions: pd.DataFrame, split_config: SplitConfig) -> Dict[str, pd.DataFrame]:
    """Global time-based split.

    This is simple and interview-friendly. A per-user temporal split can be added later.
    """
    total_frac = split_config.train_frac + split_config.val_frac + split_config.test_frac
    if not np.isclose(total_frac, 1.0):
        raise ValueError(
            f"Split fractions must sum to 1.0, got train+val+test={total_frac:.4f}."
        )

    interactions = interactions.sort_values("timestamp").reset_index(drop=True)
    n = len(interactions)
    train_end = int(n * split_config.train_frac)
    val_end = int(n * (split_config.train_frac + split_config.val_frac))

    train_df = interactions.iloc[:train_end].copy()
    val_df = interactions.iloc[train_end:val_end].copy()
    test_df = interactions.iloc[val_end:].copy()

    return {"train": train_df, "val": val_df, "test": test_df}


def build_eda_summary(interactions: pd.DataFrame, movies: pd.DataFrame) -> Dict[str, object]:
    num_users = interactions["userId"].nunique()
    num_items = interactions["movieId"].nunique()
    num_interactions = len(interactions)
    sparsity = 1.0 - (num_interactions / max(num_users * num_items, 1))
    merged = interactions.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    genre_counts = (
        merged.assign(genres=merged["genres"].fillna("(no genres listed)").str.split("|"))
        .explode("genres")
        .groupby("genres")["movieId"]
        .count()
        .sort_values(ascending=False)
    )
    popular_movies = (
        merged.groupby(["movieId", "title"])["userId"]
        .count()
        .reset_index(name="interaction_count")
        .sort_values("interaction_count", ascending=False)
        .head(10)
    )

    summary = {
        "num_users": int(num_users),
        "num_items": int(num_items),
        "num_interactions": int(num_interactions),
        "sparsity": float(round(sparsity, 6)),
        "avg_interactions_per_user": float(round(num_interactions / max(num_users, 1), 3)),
        "avg_interactions_per_item": float(round(num_interactions / max(num_items, 1), 3)),
        "top_genres": genre_counts.head(10).to_dict(),
        "top_movies": popular_movies.to_dict(orient="records"),
    }
    return summary


def attach_movie_metadata(interactions: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    return interactions.merge(movies, on="movieId", how="left")
