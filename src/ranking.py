from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from catboost import CatBoostRanker, Pool

from src.config import RankingConfig


FEATURE_COLUMNS = [
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
]

CATEGORICAL_COLUMNS = []


@dataclass
class RankerArtifacts:
    model: CatBoostRanker
    feature_columns: List[str]


def train_ranker(train_candidates: pd.DataFrame, val_candidates: pd.DataFrame, config: RankingConfig) -> RankerArtifacts:
    train_pool = Pool(
        data=train_candidates[FEATURE_COLUMNS],
        label=train_candidates["label"],
        group_id=train_candidates["userId"],
        cat_features=CATEGORICAL_COLUMNS,
    )
    val_pool = Pool(
        data=val_candidates[FEATURE_COLUMNS],
        label=val_candidates["label"],
        group_id=val_candidates["userId"],
        cat_features=CATEGORICAL_COLUMNS,
    )

    model = CatBoostRanker(
        iterations=config.catboost_iterations,
        depth=config.catboost_depth,
        learning_rate=config.catboost_learning_rate,
        loss_function="YetiRank",
        eval_metric="NDCG",
        random_seed=config.random_seed,
        verbose=50,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return RankerArtifacts(model=model, feature_columns=FEATURE_COLUMNS)


def score_candidates(dataset: pd.DataFrame, artifacts: RankerArtifacts) -> pd.DataFrame:
    out = dataset.copy()
    out["rank_score"] = artifacts.model.predict(out[artifacts.feature_columns])
    out = out.sort_values(["userId", "rank_score"], ascending=[True, False])
    return out
