from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


def recall_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    actual_set = set(actual)
    if not actual_set:
        return 0.0
    pred_set = set(predicted[:k])
    return len(actual_set & pred_set) / len(actual_set)


def apk(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    actual_set = set(actual)
    if not actual_set:
        return 0.0

    score = 0.0
    hits = 0.0
    used = set()
    for i, p in enumerate(predicted[:k], start=1):
        if p in actual_set and p not in used:
            hits += 1.0
            score += hits / i
            used.add(p)
    return score / min(len(actual_set), k)


def ndcg_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    actual_set = set(actual)
    dcg = 0.0
    for i, p in enumerate(predicted[:k], start=1):
        if p in actual_set:
            dcg += 1.0 / np.log2(i + 1)
    ideal_hits = min(len(actual_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def evaluate_topk(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int = 10,
    score_col: str = "rank_score",
) -> Dict[str, float]:
    gt_map = ground_truth.groupby("userId")["movieId"].apply(list).to_dict()

    metrics = {"recall@k": [], "map@k": [], "ndcg@k": []}

    for user_id, group in predictions.sort_values(["userId", score_col], ascending=[True, False]).groupby("userId"):
        actual = gt_map.get(user_id, [])
        if not actual:
            continue
        predicted = group["movieId"].tolist()
        metrics["recall@k"].append(recall_at_k(actual, predicted, k=k))
        metrics["map@k"].append(apk(actual, predicted, k=k))
        metrics["ndcg@k"].append(ndcg_at_k(actual, predicted, k=k))

    return {name: float(np.mean(values)) if values else 0.0 for name, values in metrics.items()}


def build_predictions_from_ranked_items(
    ranked_items_by_user: Dict[int, Sequence[int]],
    score_col: str = "rank_score",
) -> pd.DataFrame:
    rows = []
    for user_id, items in ranked_items_by_user.items():
        for rank, movie_id in enumerate(items, start=1):
            rows.append({"userId": user_id, "movieId": movie_id, score_col: float(-rank)})
    return pd.DataFrame(rows)


def persist_offline_run_artifacts(
    report: dict,
    artifacts_dir: Path,
    *,
    data_dir: str | None = None,
) -> Path:
    """Сохраняет копию отчёта о метриках в artifacts/runs/<run_id>/."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(artifacts_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["run_id"] = run_id
    if data_dir is not None:
        payload["data_dir"] = data_dir
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    k = report.get("k", "?")
    lines = [
        f"# Offline metrics (k={k})",
        "",
        f"run_id: {run_id}",
        "",
    ]
    for name, m in report.items():
        if name in ("k", "run_id") or not isinstance(m, dict):
            continue
        lines.append(f"## {name}")
        for metric_name, metric_value in m.items():
            lines.append(f"- {metric_name}: {metric_value:.6f}")
        lines.append("")
    (run_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    (Path(artifacts_dir) / "latest_run.txt").write_text(run_id + "\n", encoding="utf-8")
    return run_dir
