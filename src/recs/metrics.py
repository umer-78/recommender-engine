"""Ranking measures for a recommender.

All of these take a ranked list and the set of items the user actually went on to
interact with. They are top-k measures: a recommender is judged on the handful of
slots it actually gets, not on how it orders the other ten thousand items nobody
will scroll to.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Of the k slots shown, what share were hits.

    The denominator is k, not len(recommended). Shorter lists would otherwise be
    flattered: a recommender that returns one correct item and gives up scores
    1.0 against a full page that got four out of ten right.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    top = recommended[:k]
    return sum(1 for item in top if item in relevant) / k


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Of everything the user went on to like, what share made the top k."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant)


def average_precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Precision averaged at each hit — rewards putting the hits near the top.

    Divided by min(k, |relevant|): a user with two relevant items cannot fill ten
    slots, and scoring them out of ten would call a perfect page a 20%.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant:
        return 0.0

    hits = 0
    total = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / min(k, len(relevant))


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Discounted cumulative gain, normalised by the best possible ordering.

    Binary relevance, log2 discount. 1.0 means every relevant item the user had
    was placed as high as it could have been.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant:
        return 0.0

    gain = sum(1 / math.log2(rank + 1)
               for rank, item in enumerate(recommended[:k], start=1) if item in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return gain / ideal if ideal else 0.0


def hit_rate(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """1.0 if the page contains anything the user wanted, else 0.0."""
    if k < 1:
        raise ValueError("k must be at least 1")
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def catalogue_coverage(all_recommendations: Sequence[Sequence[int]], catalogue_size: int,
                       k: int) -> float:
    """Share of the catalogue that appears in anyone's top k.

    A recommender that shows the same twenty blockbusters to everyone can score
    well on accuracy and still be useless. This is the number that catches it.
    """
    if catalogue_size < 1:
        raise ValueError("the catalogue cannot be empty")
    shown = {item for row in all_recommendations for item in row[:k]}
    return len(shown) / catalogue_size


def novelty(all_recommendations: Sequence[Sequence[int]], popularity: dict[int, int],
            total_interactions: int, k: int) -> float:
    """Mean self-information of what was recommended, in bits.

    Higher means the recommender is surfacing less obvious items. Popular items
    carry little information; a recommendation nobody has seen carries a lot.
    """
    if total_interactions < 1:
        raise ValueError("no interactions to measure popularity against")

    scores = []
    for row in all_recommendations:
        for item in row[:k]:
            count = popularity.get(item, 0)
            probability = count / total_interactions if count else 1 / total_interactions
            scores.append(-math.log2(probability))
    return sum(scores) / len(scores) if scores else 0.0
