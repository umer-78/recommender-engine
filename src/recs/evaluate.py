"""Scoring recommenders on the same users, folds and metrics."""

from __future__ import annotations

from dataclasses import dataclass

from .data import Dataset, Split
from .metrics import (
    average_precision_at_k,
    catalogue_coverage,
    hit_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .models import Recommender


@dataclass(frozen=True)
class Score:
    model: str
    k: int
    users: int
    precision: float
    recall: float
    map: float
    ndcg: float
    hit_rate: float
    coverage: float

    def row(self) -> str:
        return (f"{self.model:<22}{self.precision:>10.4f}{self.recall:>10.4f}"
                f"{self.map:>10.4f}{self.ndcg:>10.4f}{self.hit_rate:>9.3f}{self.coverage:>10.3f}")


def evaluate(model: Recommender, split: Split, full: Dataset, k: int = 10) -> Score:
    """Fit on the training half and score the held-out items for every user."""
    model.fit(split.train)
    users = split.evaluable_users

    pages = []
    precision = recall = average_precision = ndcg = hits = 0.0
    for user in users:
        relevant = split.test[user]
        page = model.recommend(user, k)
        pages.append(page)
        precision += precision_at_k(page, relevant, k)
        recall += recall_at_k(page, relevant, k)
        average_precision += average_precision_at_k(page, relevant, k)
        ndcg += ndcg_at_k(page, relevant, k)
        hits += hit_rate(page, relevant, k)

    n = len(users)
    return Score(
        model=model.name, k=k, users=n,
        precision=precision / n, recall=recall / n, map=average_precision / n,
        ndcg=ndcg / n, hit_rate=hits / n,
        coverage=catalogue_coverage(pages, len(full.items), k),
    )


def leaderboard(models: list[Recommender], split: Split, full: Dataset, k: int = 10) -> list[Score]:
    """Every model on identical users and identical held-out items, best NDCG first."""
    return sorted((evaluate(m, split, full, k) for m in models), key=lambda s: -s.ndcg)


def header(k: int) -> str:
    return (f"{'model':<22}{f'P@{k}':>10}{f'R@{k}':>10}{f'MAP@{k}':>10}"
            f"{f'NDCG@{k}':>10}{'hit':>9}{'coverage':>10}")
