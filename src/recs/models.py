"""Recommenders, baselines first.

Every model here returns a ranked list of items a user has *not* already
interacted with. Leaving seen items in is the oldest bug in recommendation: the
scores look wonderful, because the best predictor of what someone watched is
what they watched, and the product recommends the film they finished last night.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from .data import Dataset


class Recommender:
    name = "recommender"

    def fit(self, data: Dataset) -> Recommender:
        raise NotImplementedError

    def recommend(self, user: int, k: int = 10) -> list[int]:
        raise NotImplementedError

    def _unseen(self, user: int, ranked: Sequence[tuple[int, float]], k: int) -> list[int]:
        seen = self._seen.get(user, set())
        out = []
        for item, _score in ranked:
            if item in seen:
                continue
            out.append(item)
            if len(out) == k:
                break
        return out

    def _remember(self, data: Dataset) -> None:
        seen: dict[int, set[int]] = defaultdict(set)
        for interaction in data.interactions:
            seen[interaction.user].add(interaction.item)
        self._seen = dict(seen)


@dataclass
class Popularity(Recommender):
    """The most interacted-with items, same list for everybody.

    Kept as a first-class model because it is a genuinely strong baseline and
    most published comparisons quietly omit it. If a personalised model cannot
    beat this, it is costing compute to be worse than a bestseller shelf.
    """

    name: str = "popularity"
    _ranked: list[tuple[int, float]] = field(default_factory=list)

    def fit(self, data: Dataset) -> Popularity:
        counts = data.popularity()
        self._ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self._remember(data)
        return self

    def recommend(self, user: int, k: int = 10) -> list[int]:
        return self._unseen(user, self._ranked, k)


@dataclass
class RandomRecommender(Recommender):
    """A shuffled catalogue. The floor: anything at or below this is noise."""

    seed: int = 7
    name: str = "random"
    _items: list[int] = field(default_factory=list)

    def fit(self, data: Dataset) -> RandomRecommender:
        self._items = sorted(data.items)
        self._remember(data)
        return self

    def recommend(self, user: int, k: int = 10) -> list[int]:
        rng = random.Random(self.seed + user)
        shuffled = self._items[:]
        rng.shuffle(shuffled)
        return self._unseen(user, [(item, 0.0) for item in shuffled], k)


@dataclass
class ItemKNN(Recommender):
    """Item-item collaborative filtering on cosine similarity.

    "People who liked this also liked that." Similarity is shrunk toward zero
    when two items share few users: three co-occurrences can produce a cosine of
    1.0, and without shrinkage those accidents dominate the neighbourhood of
    every obscure item.
    """

    neighbours: int = 50
    shrinkage: float = 20.0
    name: str = "item-knn"
    _similar: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    _history: dict[int, dict[int, float]] = field(default_factory=dict)

    def fit(self, data: Dataset) -> ItemKNN:
        by_item: dict[int, dict[int, float]] = defaultdict(dict)
        by_user: dict[int, dict[int, float]] = defaultdict(dict)
        for interaction in data.interactions:
            by_item[interaction.item][interaction.user] = interaction.rating
            by_user[interaction.user][interaction.item] = interaction.rating

        norms = {item: math.sqrt(sum(v * v for v in users.values()))
                 for item, users in by_item.items()}

        # Only item pairs that share a user can have a non-zero cosine, so walk
        # each user's basket instead of every pair in the catalogue.
        dot: dict[tuple[int, int], float] = defaultdict(float)
        shared: dict[tuple[int, int], int] = defaultdict(int)
        for items in by_user.values():
            entries = sorted(items.items())
            for i, (a, ra) in enumerate(entries):
                for b, rb in entries[i + 1:]:
                    dot[(a, b)] += ra * rb
                    shared[(a, b)] += 1

        neighbourhood: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (a, b), product in dot.items():
            denominator = norms[a] * norms[b]
            if not denominator:
                continue
            count = shared[(a, b)]
            # Shrinkage: count / (count + shrinkage) -> 0 for rare co-occurrence.
            similarity = product / denominator * (count / (count + self.shrinkage))
            neighbourhood[a].append((b, similarity))
            neighbourhood[b].append((a, similarity))

        self._similar = {
            item: sorted(neighbours, key=lambda kv: -kv[1])[:self.neighbours]
            for item, neighbours in neighbourhood.items()
        }
        self._history = dict(by_user)
        self._remember(data)
        return self

    def recommend(self, user: int, k: int = 10) -> list[int]:
        history = self._history.get(user)
        if not history:
            return []

        scores: dict[int, float] = defaultdict(float)
        for item, rating in history.items():
            for neighbour, similarity in self._similar.get(item, ()):
                scores[neighbour] += similarity * rating

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return self._unseen(user, ranked, k)


@dataclass
class MatrixFactorization(Recommender):
    """Latent factors learned by SGD, with a global mean and per-user/item biases.

    The biases carry most of the signal in practice — some users rate everything
    highly, some items are simply better — and learning them separately stops the
    factors from spending their capacity re-deriving that.
    """

    factors: int = 24
    epochs: int = 30
    learning_rate: float = 0.01
    regularisation: float = 0.08
    seed: int = 11
    name: str = "matrix factorisation"
    _mean: float = 0.0
    _user_bias: dict[int, float] = field(default_factory=dict)
    _item_bias: dict[int, float] = field(default_factory=dict)
    _user_factors: dict[int, list[float]] = field(default_factory=dict)
    _item_factors: dict[int, list[float]] = field(default_factory=dict)
    _items: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)

    def fit(self, data: Dataset) -> MatrixFactorization:
        rng = random.Random(self.seed)
        rows = list(data.interactions)
        self._mean = sum(i.rating for i in rows) / len(rows)
        self._items = sorted(data.items)

        spread = 0.1
        self._user_bias = dict.fromkeys(data.users, 0.0)
        self._item_bias = dict.fromkeys(self._items, 0.0)
        self._user_factors = {u: [rng.gauss(0, spread) for _ in range(self.factors)]
                              for u in data.users}
        self._item_factors = {i: [rng.gauss(0, spread) for _ in range(self.factors)]
                              for i in self._items}

        self.losses = []
        for _ in range(self.epochs):
            rng.shuffle(rows)
            squared = 0.0
            for row in rows:
                u, i = row.user, row.item
                pu, qi = self._user_factors[u], self._item_factors[i]
                prediction = (self._mean + self._user_bias[u] + self._item_bias[i]
                              + sum(a * b for a, b in zip(pu, qi, strict=True)))
                error = row.rating - prediction
                squared += error * error

                self._user_bias[u] += self.learning_rate * (error - self.regularisation * self._user_bias[u])
                self._item_bias[i] += self.learning_rate * (error - self.regularisation * self._item_bias[i])
                for f in range(self.factors):
                    pu_f, qi_f = pu[f], qi[f]
                    pu[f] += self.learning_rate * (error * qi_f - self.regularisation * pu_f)
                    qi[f] += self.learning_rate * (error * pu_f - self.regularisation * qi_f)
            self.losses.append(math.sqrt(squared / len(rows)))

        self._remember(data)
        return self

    def predict(self, user: int, item: int) -> float:
        score = self._mean + self._user_bias.get(user, 0.0) + self._item_bias.get(item, 0.0)
        pu, qi = self._user_factors.get(user), self._item_factors.get(item)
        if pu and qi:
            score += sum(a * b for a, b in zip(pu, qi, strict=True))
        return score

    def recommend(self, user: int, k: int = 10) -> list[int]:
        if user not in self._user_factors:
            return []
        ranked = sorted(((item, self.predict(user, item)) for item in self._items),
                        key=lambda kv: (-kv[1], kv[0]))
        return self._unseen(user, ranked, k)


@dataclass
class BPR(Recommender):
    """Matrix factorisation trained on a *ranking* loss instead of a rating one.

    Rating prediction and ranking are not the same task, and optimising the first
    does not optimise the second. `MatrixFactorization` minimises squared error on
    observed ratings, which pushes the highest predicted scores toward items that
    a handful of generous users rated well — items nobody else wants. Ranked by
    that score it lands below a random shuffle on this data.

    BPR (Bayesian Personalised Ranking) drops the ratings entirely and learns from
    pairs: for each thing a user interacted with, sample something they did not,
    and push the first above the second. The gradient is of
    -log sigmoid(score_positive - score_negative), so the only thing being
    optimised is the order — which is the only thing the metrics measure.
    """

    factors: int = 32
    epochs: int = 40
    learning_rate: float = 0.05
    regularisation: float = 0.01
    seed: int = 13
    name: str = "bpr"
    _user_factors: dict[int, list[float]] = field(default_factory=dict)
    _item_factors: dict[int, list[float]] = field(default_factory=dict)
    _item_bias: dict[int, float] = field(default_factory=dict)
    _items: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)

    def fit(self, data: Dataset) -> BPR:
        rng = random.Random(self.seed)
        self._items = sorted(data.items)
        positives: dict[int, set[int]] = defaultdict(set)
        for interaction in data.interactions:
            positives[interaction.user].add(interaction.item)

        spread = 0.1
        self._user_factors = {u: [rng.gauss(0, spread) for _ in range(self.factors)]
                              for u in data.users}
        self._item_factors = {i: [rng.gauss(0, spread) for _ in range(self.factors)]
                              for i in self._items}
        self._item_bias = dict.fromkeys(self._items, 0.0)

        pairs = [(i.user, i.item) for i in data.interactions]
        item_count = len(self._items)

        self.losses = []
        for _ in range(self.epochs):
            rng.shuffle(pairs)
            total = 0.0
            for user, positive in pairs:
                seen = positives[user]
                if len(seen) >= item_count:
                    continue
                negative = self._items[rng.randrange(item_count)]
                while negative in seen:
                    negative = self._items[rng.randrange(item_count)]

                pu = self._user_factors[user]
                qi, qj = self._item_factors[positive], self._item_factors[negative]
                difference = (self._item_bias[positive] - self._item_bias[negative]
                              + sum(pu[f] * (qi[f] - qj[f]) for f in range(self.factors)))
                # sigmoid(-x); written this way so a large positive difference
                # cannot overflow exp().
                weight = 1 / (1 + math.exp(difference)) if difference > -30 else 1.0
                total += math.log1p(math.exp(-abs(difference))) + max(-difference, 0)

                self._item_bias[positive] += self.learning_rate * (
                    weight - self.regularisation * self._item_bias[positive])
                self._item_bias[negative] += self.learning_rate * (
                    -weight - self.regularisation * self._item_bias[negative])
                for f in range(self.factors):
                    pu_f, qi_f, qj_f = pu[f], qi[f], qj[f]
                    pu[f] += self.learning_rate * (weight * (qi_f - qj_f) - self.regularisation * pu_f)
                    qi[f] += self.learning_rate * (weight * pu_f - self.regularisation * qi_f)
                    qj[f] += self.learning_rate * (-weight * pu_f - self.regularisation * qj_f)
            self.losses.append(total / max(len(pairs), 1))

        self._remember(data)
        return self

    def score(self, user: int, item: int) -> float:
        pu, qi = self._user_factors.get(user), self._item_factors.get(item)
        if not pu or not qi:
            return float("-inf")
        return self._item_bias[item] + sum(a * b for a, b in zip(pu, qi, strict=True))

    def recommend(self, user: int, k: int = 10) -> list[int]:
        if user not in self._user_factors:
            return []
        ranked = sorted(((item, self.score(user, item)) for item in self._items),
                        key=lambda kv: (-kv[1], kv[0]))
        return self._unseen(user, ranked, k)


def default_models() -> list[Recommender]:
    return [
        RandomRecommender(),
        Popularity(),
        ItemKNN(neighbours=50, shrinkage=20),
        MatrixFactorization(factors=24, epochs=30),
        BPR(factors=32, epochs=40),
    ]
