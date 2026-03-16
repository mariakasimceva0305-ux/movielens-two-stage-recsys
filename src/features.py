from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _explode_genres(movies: pd.DataFrame) -> pd.DataFrame:
    out = movies[["movieId", "genres"]].copy()
    out["genres"] = out["genres"].fillna("(no genres listed)").str.split("|")
    out = out.explode("genres")
    return out


def build_user_features(train_df: pd.DataFrame) -> pd.DataFrame:
    user_stats = (
        train_df.groupby("userId")
        .agg(
            user_interaction_count=("movieId", "count"),
            user_first_ts=("timestamp", "min"),
            user_last_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    user_stats["user_active_span_days"] = (
        (user_stats["user_last_ts"] - user_stats["user_first_ts"]) / 86400.0
    )
    user_stats["user_recency_hours"] = (train_df["timestamp"].max() - user_stats["user_last_ts"]) / 3600.0
    return user_stats


def build_item_features(train_df: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    item_stats = (
        train_df.groupby("movieId")
        .agg(
            item_interaction_count=("userId", "count"),
            item_unique_users=("userId", "nunique"),
            item_first_ts=("timestamp", "min"),
            item_last_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    item_stats["item_popularity_rank"] = item_stats["item_interaction_count"].rank(method="average", ascending=False)
    item_stats["item_popularity_pct"] = item_stats["item_interaction_count"].rank(pct=True)
    item_stats["item_recency_hours"] = (train_df["timestamp"].max() - item_stats["item_last_ts"]) / 3600.0

    return item_stats.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")


def build_user_genre_affinity(train_df: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    interactions = train_df.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    interactions["genres"] = interactions["genres"].fillna("(no genres listed)").str.split("|")
    exploded = interactions.explode("genres")

    affinity = (
        exploded.groupby(["userId", "genres"])['movieId']
        .count()
        .reset_index(name="user_genre_count")
    )
    total = affinity.groupby("userId")["user_genre_count"].transform("sum")
    affinity["user_genre_affinity"] = affinity["user_genre_count"] / total
    return affinity


def build_ranking_dataset(
    candidates_df: pd.DataFrame,
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    user_features = build_user_features(train_df)
    item_features = build_item_features(train_df, movies)
    user_genre_affinity = build_user_genre_affinity(train_df, movies)

    target_pairs = target_df[["userId", "movieId"]].drop_duplicates().assign(label=1)
    dataset = candidates_df.merge(target_pairs, on=["userId", "movieId"], how="left")
    dataset["label"] = dataset["label"].fillna(0).astype(int)

    dataset = dataset.merge(user_features, on="userId", how="left")
    dataset = dataset.merge(item_features, on="movieId", how="left")

    user_item_history = (
        train_df.groupby(["userId", "movieId"])
        .agg(
            user_item_interaction_count=("timestamp", "count"),
            user_item_last_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    dataset = dataset.merge(user_item_history, on=["userId", "movieId"], how="left")

    genre_map = movies[["movieId", "genres"]].copy()
    genre_map["genres"] = genre_map["genres"].fillna("(no genres listed)").str.split("|")
    genre_map = genre_map.explode("genres")

    dataset = dataset.merge(genre_map, on="movieId", how="left")
    dataset = dataset.merge(user_genre_affinity, on=["userId", "genres"], how="left")
    dataset["user_genre_affinity"] = dataset["user_genre_affinity"].fillna(0.0)

    dataset["candidate_source_is_item_item"] = (dataset["candidate_source"] == "item_item").astype(int)
    train_end_ts = train_df["timestamp"].max()
    dataset["hours_since_item_last_seen"] = (train_end_ts - dataset["item_last_ts"]) / 3600.0
    dataset["hours_since_user_last_seen"] = (train_end_ts - dataset["user_last_ts"]) / 3600.0
    dataset["hours_since_user_item_seen"] = (train_end_ts - dataset["user_item_last_ts"]) / 3600.0

    # Collapse back to one row per user-item.
    numeric_cols = [
        "candidate_rank",
        "item_item_score",
        "user_interaction_count",
        "user_active_span_days",
        "user_recency_hours",
        "item_interaction_count",
        "item_unique_users",
        "item_popularity_rank",
        "item_popularity_pct",
        "item_recency_hours",
        "candidate_source_is_item_item",
        "hours_since_item_last_seen",
        "hours_since_user_last_seen",
        "hours_since_user_item_seen",
        "user_item_interaction_count",
        "user_genre_affinity",
        "label",
    ]
    agg_map = {col: "max" for col in numeric_cols}
    agg_map.update({"title": "first", "genres": "first", "candidate_source": "first"})

    dataset = dataset.groupby(["userId", "movieId"], as_index=False).agg(agg_map)

    fill_zero_cols = [
        "item_item_score",
        "user_interaction_count",
        "user_active_span_days",
        "user_recency_hours",
        "item_interaction_count",
        "item_unique_users",
        "item_popularity_rank",
        "item_popularity_pct",
        "item_recency_hours",
        "candidate_source_is_item_item",
        "hours_since_item_last_seen",
        "hours_since_user_last_seen",
        "hours_since_user_item_seen",
        "user_item_interaction_count",
        "user_genre_affinity",
    ]
    for col in fill_zero_cols:
        if col in dataset.columns:
            dataset[col] = dataset[col].fillna(0.0)

    return dataset
