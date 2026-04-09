from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.candidate_generation import (
    ItemItemRecommender,
    PopularityRecommender,
    build_seen_items_map,
    generate_candidate_frame,
)
from src.config import ArtifactConfig, CandidateConfig, DataConfig, RankingConfig, SplitConfig
from src.data import build_eda_summary, convert_to_implicit, filter_active_users, load_movielens, time_based_split
from src.evaluate import build_predictions_from_ranked_items, evaluate_topk, persist_offline_run_artifacts
from src.features import build_ranking_dataset
from src.ranking import score_candidates, train_ranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MovieLens two-stage recommendation pipeline")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--top-k-candidates", type=int, default=100)
    parser.add_argument("--k-metrics", type=int, default=10)
    return parser.parse_args()


def _split_users_for_ranker(val_df):
    users = sorted(val_df["userId"].unique().tolist())
    if not users:
        return set(), set()
    cut = max(1, int(len(users) * 0.8))
    train_users = set(users[:cut])
    eval_users = set(users[cut:]) or train_users
    return train_users, eval_users


def _eval_popularity(popularity_model, seen_items_map, users, ground_truth, k):
    ranked_items_by_user = {
        int(user_id): popularity_model.recommend(seen_items=seen_items_map.get(int(user_id), set()), k=k)
        for user_id in users
    }
    pred_df = build_predictions_from_ranked_items(ranked_items_by_user, score_col="baseline_score")
    return evaluate_topk(pred_df, ground_truth=ground_truth, k=k, score_col="baseline_score")


def _eval_item_item(item_item_model, users, ground_truth, k):
    ranked_items_by_user = {int(user_id): item_item_model.recommend(int(user_id), k=k) for user_id in users}
    pred_df = build_predictions_from_ranked_items(ranked_items_by_user, score_col="baseline_score")
    return evaluate_topk(pred_df, ground_truth=ground_truth, k=k, score_col="baseline_score")


def main() -> None:
    args = parse_args()

    data_config = DataConfig(data_dir=Path(args.data_dir), min_rating=args.min_rating)
    split_config = SplitConfig()
    candidate_config = CandidateConfig(top_k_candidates=args.top_k_candidates)
    ranking_config = RankingConfig()
    artifact_config = ArtifactConfig(artifacts_dir=Path(args.artifacts_dir))
    artifact_config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    ratings, movies = load_movielens(data_config)
    implicit_df = convert_to_implicit(ratings, min_rating=data_config.min_rating)
    implicit_df = filter_active_users(implicit_df, min_user_interactions=data_config.min_user_interactions)

    eda_summary = build_eda_summary(implicit_df, movies)
    print("EDA summary:")
    for k, v in eda_summary.items():
        print(f"  {k}: {v}")

    splits = time_based_split(implicit_df, split_config)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    popularity_model = PopularityRecommender.fit(train_df)
    item_item_model = ItemItemRecommender.fit(train_df)
    seen_items_map = build_seen_items_map(train_df)

    val_users = val_df["userId"].unique().tolist()
    test_users = test_df["userId"].unique().tolist()

    val_candidates = generate_candidate_frame(
        users=val_users,
        seen_items_map=seen_items_map,
        popularity_model=popularity_model,
        item_item_model=item_item_model,
        top_k_candidates=candidate_config.top_k_candidates,
    )
    test_candidates = generate_candidate_frame(
        users=test_users,
        seen_items_map=seen_items_map,
        popularity_model=popularity_model,
        item_item_model=item_item_model,
        top_k_candidates=candidate_config.top_k_candidates,
    )

    ranking_source_dataset = build_ranking_dataset(val_candidates, train_df, val_df, movies)
    rank_train_users, rank_eval_users = _split_users_for_ranker(val_df)
    rank_train_dataset = ranking_source_dataset[ranking_source_dataset["userId"].isin(rank_train_users)].copy()
    rank_eval_dataset = ranking_source_dataset[ranking_source_dataset["userId"].isin(rank_eval_users)].copy()

    if rank_train_dataset.empty or rank_eval_dataset.empty:
        raise ValueError(
            "Ranking train/eval datasets are empty. Check split sizes or reduce filtering constraints."
        )

    test_dataset = build_ranking_dataset(test_candidates, train_df, test_df, movies)

    ranker_artifacts = train_ranker(rank_train_dataset, rank_eval_dataset, ranking_config)

    scored_test = score_candidates(test_dataset, ranker_artifacts)
    two_stage_metrics = evaluate_topk(scored_test, ground_truth=test_df, k=args.k_metrics)

    popularity_metrics = _eval_popularity(
        popularity_model=popularity_model,
        seen_items_map=seen_items_map,
        users=test_users,
        ground_truth=test_df,
        k=args.k_metrics,
    )
    item_item_metrics = _eval_item_item(
        item_item_model=item_item_model,
        users=test_users,
        ground_truth=test_df,
        k=args.k_metrics,
    )

    metrics_report = {
        "k": args.k_metrics,
        "popularity": popularity_metrics,
        "item_item": item_item_metrics,
        "two_stage": two_stage_metrics,
    }

    print(f"\nOffline metrics @ {args.k_metrics}:")
    for model_name, model_metrics in metrics_report.items():
        if model_name == "k":
            continue
        print(f"\n{model_name}:")
        for metric_name, metric_value in model_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

    joblib.dump(popularity_model, artifact_config.artifacts_dir / "popularity_model.joblib")
    joblib.dump(item_item_model, artifact_config.artifacts_dir / "item_item_model.joblib")
    joblib.dump(ranker_artifacts, artifact_config.artifacts_dir / "ranker_artifacts.joblib")
    joblib.dump(seen_items_map, artifact_config.artifacts_dir / "seen_items_map.joblib")
    joblib.dump(train_df, artifact_config.artifacts_dir / "train_df.joblib")
    joblib.dump(movies, artifact_config.artifacts_dir / "movies_df.joblib")
    with open(artifact_config.artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, ensure_ascii=False, indent=2)

    persist_offline_run_artifacts(
        metrics_report,
        artifact_config.artifacts_dir,
        data_dir=str(args.data_dir),
    )

    print(f"\nArtifacts saved to: {artifact_config.artifacts_dir}")


if __name__ == "__main__":
    main()
