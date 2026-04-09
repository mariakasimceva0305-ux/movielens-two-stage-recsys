from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from src.inference import InferenceBundle, recommend_for_user


def _minimal_bundle() -> InferenceBundle:
    pop = MagicMock()
    pop.recommend = MagicMock(side_effect=lambda seen_items, k: [101, 102, 103][:k])
    ii = MagicMock()
    ii.recommend = MagicMock(return_value=[201, 202])
    ranker = MagicMock()
    movies = pd.DataFrame(
        {
            "movieId": [101, 102, 103, 201, 202],
            "title": ["A", "B", "C", "D", "E"],
        }
    )
    train_df = pd.DataFrame({"userId": [1], "movieId": [1], "rating": [1.0]})
    seen = {1: {1}}
    return InferenceBundle(
        popularity_model=pop,
        item_item_model=ii,
        ranker_artifacts=ranker,
        seen_items_map=seen,
        train_df=train_df,
        movies_df=movies,
    )


def test_unknown_user_does_not_crash():
    b = _minimal_bundle()
    out = recommend_for_user(b, user_id=999_999, k=2)
    assert len(out) <= 2
    for x in out:
        assert "movie_id" in x
        assert "title" in x
