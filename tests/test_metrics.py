import math

import pytest

from recs import (
    average_precision_at_k,
    catalogue_coverage,
    hit_rate,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
)


def test_precision_divides_by_k_not_by_the_list_length():
    """A short list must not be flattered.

    One correct item out of one shown is not better than four out of ten; both
    pages had ten slots and one of them used them.
    """
    assert precision_at_k([7], {7}, k=10) == 0.1
    assert precision_at_k([7, 1, 2, 3, 8, 4, 5, 9, 6, 0], {7, 8, 9, 0}, k=10) == 0.4


def test_precision_and_recall_on_hand_worked_numbers():
    page, relevant = [1, 2, 3, 4, 5], {2, 5, 9}

    assert precision_at_k(page, relevant, 5) == 2 / 5
    assert recall_at_k(page, relevant, 5) == pytest.approx(2 / 3)


def test_average_precision_rewards_hits_near_the_top():
    relevant = {1, 2}
    top = average_precision_at_k([1, 2, 9, 8, 7], relevant, 5)
    bottom = average_precision_at_k([9, 8, 7, 1, 2], relevant, 5)

    assert top == 1.0
    assert bottom < top


def test_average_precision_is_divided_by_what_was_achievable():
    """Two relevant items cannot fill ten slots; scoring them out of ten would
    call a flawless page a 20%."""
    assert average_precision_at_k([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], {1, 2}, 10) == 1.0


def test_ndcg_is_one_for_the_best_possible_ordering():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 5) == 1.0
    assert ndcg_at_k([9, 8, 1, 2, 3], {1, 2, 3}, 5) < 1.0


def test_ndcg_uses_a_log2_discount():
    # One hit at rank 2: gain 1/log2(3), ideal 1/log2(2) = 1.
    assert ndcg_at_k([9, 1], {1}, 2) == pytest.approx(1 / math.log2(3))


def test_every_measure_is_zero_when_nothing_is_relevant():
    for fn in (precision_at_k, recall_at_k, average_precision_at_k, ndcg_at_k, hit_rate):
        assert fn([1, 2, 3], set(), 3) == 0.0


def test_hit_rate_is_binary():
    assert hit_rate([1, 2, 3], {3}, 3) == 1.0
    assert hit_rate([1, 2, 3], {4}, 3) == 0.0


def test_metrics_reject_a_meaningless_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        precision_at_k([1], {1}, 0)


def test_coverage_counts_distinct_items_shown_to_anyone():
    # Three pages, four distinct items, catalogue of ten.
    pages = [[1, 2], [2, 3], [4, 1]]
    assert catalogue_coverage(pages, catalogue_size=10, k=2) == 0.4


def test_novelty_is_higher_for_less_popular_items():
    popularity = {1: 900, 2: 50, 3: 1}

    obvious = novelty([[1]], popularity, total_interactions=1000, k=1)
    obscure = novelty([[3]], popularity, total_interactions=1000, k=1)

    assert obscure > obvious
    assert obvious == pytest.approx(-math.log2(0.9))
