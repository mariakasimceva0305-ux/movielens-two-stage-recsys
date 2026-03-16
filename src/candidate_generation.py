from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class PopularityRecommender:
    top_items: List[int]

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "PopularityRecommender":
        top_items = (
            train_df.groupby("movieId")["userId"]
            .count()
            .sort_values(ascending=False)
            .index.tolist()
        )
        return cls(top_items=top_items)

    def recommend(self, seen_items: Set[int], k: int = 10) -> List[int]:
        recs = [item for item in self.top_items if item not in seen_items]
        return recs[:k]


@dataclass
class ItemItemRecommender:
    item_sim_matrix: np.ndarray
    user_to_items: Dict[int, List[int]]
    movie_ids: List[int]
    movie_id_to_idx: Dict[int, int]

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "ItemItemRecommender":
        user_ids = sorted(train_df["userId"].unique())
        movie_ids = sorted(train_df["movieId"].unique())
        user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
        movie_id_to_idx = {m: i for i, m in enumerate(movie_ids)}

        rows = train_df["userId"].map(user_id_to_idx).to_numpy()
        cols = train_df["movieId"].map(movie_id_to_idx).to_numpy()
        vals = np.ones(len(train_df), dtype=np.float32)
        matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

        item_sim_matrix = cosine_similarity(matrix.T, dense_output=True)
        np.fill_diagonal(item_sim_matrix, 0.0)

        user_to_items = train_df.groupby("userId")["movieId"].apply(list).to_dict()
        return cls(
            item_sim_matrix=item_sim_matrix,
            user_to_items=user_to_items,
            movie_ids=movie_ids,
            movie_id_to_idx=movie_id_to_idx,
        )

    def score_candidates(self, user_id: int) -> Dict[int, float]:
        history = self.user_to_items.get(user_id, [])
        if not history:
            return {}

        scores = defaultdict(float)
        seen = set(history)

        for movie_id in history:
            idx = self.movie_id_to_idx.get(movie_id)
            if idx is None:
                continue
            sims = self.item_sim_matrix[idx]
            for cand_idx, sim in enumerate(sims):
                cand_movie_id = self.movie_ids[cand_idx]
                if cand_movie_id in seen or sim <= 0:
                    continue
                scores[cand_movie_id] += float(sim)

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def recommend(self, user_id: int, k: int = 10) -> List[int]:
        scores = self.score_candidates(user_id)
        return list(scores.keys())[:k]


def build_seen_items_map(interactions: pd.DataFrame) -> Dict[int, Set[int]]:
    return interactions.groupby("userId")["movieId"].apply(lambda x: set(x.tolist())).to_dict()


def generate_candidate_frame(
    users: Iterable[int],
    seen_items_map: Dict[int, Set[int]],
    popularity_model: PopularityRecommender,
    item_item_model: ItemItemRecommender,
    top_k_candidates: int = 100,
) -> pd.DataFrame:
    """Create candidate rows for ranking.

    Includes:
    - item-item candidates
    - popularity backfill
    """
    rows = []
    for user_id in users:
        seen_items = seen_items_map.get(user_id, set())
        sim_scores = item_item_model.score_candidates(user_id)
        sim_items = list(sim_scores.keys())[:top_k_candidates]

        backfill = popularity_model.recommend(seen_items=seen_items.union(sim_items), k=top_k_candidates)
        final_items = (sim_items + backfill)[:top_k_candidates]

        for rank_pos, movie_id in enumerate(final_items, start=1):
            rows.append(
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "candidate_rank": rank_pos,
                    "candidate_source": "item_item" if movie_id in sim_scores else "popularity",
                    "item_item_score": float(sim_scores.get(movie_id, 0.0)),
                }
            )

    return pd.DataFrame(rows)
