from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluate import apk, evaluate_topk, ndcg_at_k, persist_offline_run_artifacts, recall_at_k


def test_recall_at_k_simple():
    assert recall_at_k([1, 2], [2, 3, 1], k=3) == 1.0
    assert recall_at_k([10], [1, 2, 3], k=5) == 0.0


def test_apk_deterministic():
    assert apk([7, 8], [7, 9, 8], k=5) > 0.0


def test_ndcg_perfect():
    assert ndcg_at_k([1], [1, 2, 3], k=5) == 1.0


def test_evaluate_topk_aggregation():
    pred = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 11, 20, 21],
            "rank_score": [1.0, 0.5, 1.0, 0.4],
        }
    )
    gt = pd.DataFrame({"userId": [1, 2], "movieId": [10, 21]})
    m = evaluate_topk(pred, ground_truth=gt, k=2, score_col="rank_score")
    assert "recall@k" in m
    assert m["recall@k"] >= 0.5


def test_persist_offline_run_artifacts(tmp_path: Path) -> None:
    report = {"k": 10, "two_stage": {"recall@k": 0.1, "map@k": 0.2, "ndcg@k": 0.3}}
    run_dir = persist_offline_run_artifacts(report, tmp_path, data_dir="/data/ml")
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "metrics.md").is_file()
    data = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert data["run_id"]
    assert (tmp_path / "latest_run.txt").read_text(encoding="utf-8").strip() == data["run_id"]
