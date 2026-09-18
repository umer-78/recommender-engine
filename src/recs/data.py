"""Interactions in, a dataset with a temporal split out."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Interaction:
    user: int
    item: int
    rating: float
    at: datetime


@dataclass
class Dataset:
    """Interactions, plus the lookups a recommender needs."""

    interactions: list[Interaction]
    item_names: dict[int, str] = field(default_factory=dict)
    item_tags: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.interactions)

    @property
    def users(self) -> set[int]:
        return {i.user for i in self.interactions}

    @property
    def items(self) -> set[int]:
        return {i.item for i in self.interactions}

    def by_user(self) -> dict[int, list[Interaction]]:
        """Each user's history, oldest first."""
        out: dict[int, list[Interaction]] = defaultdict(list)
        for interaction in self.interactions:
            out[interaction.user].append(interaction)
        for history in out.values():
            history.sort(key=lambda i: i.at)
        return dict(out)

    def popularity(self) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for interaction in self.interactions:
            counts[interaction.item] += 1
        return dict(counts)

    def name(self, item: int) -> str:
        return self.item_names.get(item, f"item {item}")

    def subset(self, interactions: list[Interaction]) -> Dataset:
        return Dataset(interactions, self.item_names, self.item_tags)


@dataclass(frozen=True)
class Split:
    train: Dataset
    test: dict[int, set[int]]      # user -> the items they went on to interact with

    @property
    def evaluable_users(self) -> list[int]:
        return sorted(u for u, items in self.test.items() if items)


def temporal_split(data: Dataset, *, holdout: int = 2, minimum_history: int = 5) -> Split:
    """Hold out each user's last `holdout` interactions.

    Per user and by time, not at random. A random split lets a model train on
    what a user did in December and be tested on what they did in March, which is
    information no deployed recommender has. Scores from a random split are
    routinely twice what the same model achieves in production, and this is the
    single most common reason a recommender looks good in a notebook and lands
    flat.
    """
    if holdout < 1:
        raise ValueError("hold out at least one interaction")

    train: list[Interaction] = []
    test: dict[int, set[int]] = {}

    for user, history in data.by_user().items():
        if len(history) < minimum_history + holdout:
            # Too little history to both learn from and be judged on; the whole
            # of it goes to training rather than producing a meaningless score.
            train.extend(history)
            continue
        train.extend(history[:-holdout])
        test[user] = {i.item for i in history[-holdout:]}

    if not test:
        raise ValueError("no user had enough history to evaluate; lower minimum_history")

    return Split(data.subset(train), test)


def read_interactions(path: str | Path) -> list[Interaction]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            try:
                rows.append(Interaction(
                    user=int(row["user_id"]), item=int(row["item_id"]),
                    rating=float(row["rating"]),
                    at=datetime.fromisoformat(row["timestamp"]),
                ))
            except (KeyError, ValueError) as error:
                raise ValueError(f"line {line}: {error}") from None
    if not rows:
        raise ValueError(f"{path} holds no interactions")
    return rows


def read_items(path: str | Path) -> tuple[dict[int, str], dict[int, str]]:
    names: dict[int, str] = {}
    tags: dict[int, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = int(row["item_id"])
            names[item] = row["title"]
            tags[item] = row.get("genre", "")
    return names, tags


def load(directory: str | Path) -> Dataset:
    """Read interactions.csv and items.csv from a directory."""
    directory = Path(directory)
    names, tags = read_items(directory / "items.csv")
    return Dataset(read_interactions(directory / "interactions.csv"), names, tags)
