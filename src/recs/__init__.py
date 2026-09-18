"""A recommender engine: baselines, item-item CF and matrix factorisation, honestly scored."""

from .data import Dataset, Interaction, Split, load, temporal_split
from .evaluate import Score, evaluate, header, leaderboard
from .metrics import (
    average_precision_at_k,
    catalogue_coverage,
    hit_rate,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
)
from .models import (
    BPR,
    ItemKNN,
    MatrixFactorization,
    Popularity,
    RandomRecommender,
    Recommender,
    default_models,
)

__all__ = [
    "BPR", "Dataset", "Interaction", "ItemKNN", "MatrixFactorization", "Popularity",
    "RandomRecommender", "Recommender", "Score", "Split", "average_precision_at_k",
    "catalogue_coverage", "default_models", "evaluate", "header", "hit_rate", "leaderboard",
    "load", "ndcg_at_k", "novelty", "precision_at_k", "recall_at_k", "temporal_split",
]
__version__ = "1.0.0"
