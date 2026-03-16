from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: Path
    ratings_filename: str = "ratings.csv"
    movies_filename: str = "movies.csv"
    min_rating: float = 4.0
    min_user_interactions: int = 3


@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15


@dataclass
class CandidateConfig:
    top_k_popular: int = 200
    top_k_similar_per_item: int = 50
    top_k_candidates: int = 100


@dataclass
class RankingConfig:
    catboost_iterations: int = 300
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.05
    random_seed: int = 42


@dataclass
class ArtifactConfig:
    artifacts_dir: Path
