from datetime import datetime, timedelta

import pytest

from recs import Dataset, Interaction, temporal_split


def history(user: int, items: list[int], start: datetime) -> list[Interaction]:
    return [Interaction(user, item, 4.0, start + timedelta(days=i))
            for i, item in enumerate(items)]


def test_the_split_holds_out_the_latest_interactions_not_random_ones():
    """A random split lets a model train on December and be tested on March.

    That is information no deployed recommender has, and it is the single most
    common reason a model looks good in a notebook and lands flat.
    """
    start = datetime(2025, 1, 1)
    data = Dataset(history(1, [10, 20, 30, 40, 50, 60, 70], start))

    split = temporal_split(data, holdout=2, minimum_history=5)

    assert split.test[1] == {60, 70}
    assert {i.item for i in split.train.interactions} == {10, 20, 30, 40, 50}


def test_no_training_interaction_is_later_than_a_held_out_one():
    start = datetime(2025, 1, 1)
    data = Dataset(history(1, list(range(10)), start) + history(2, list(range(10, 20)), start))

    split = temporal_split(data, holdout=3, minimum_history=5)

    by_user = split.train.by_user()
    for user, held_out in split.test.items():
        latest_train = max(i.at for i in by_user[user])
        held_out_times = [i.at for i in data.by_user()[user] if i.item in held_out]
        assert latest_train < min(held_out_times), "training data leaked past the split"


def test_users_with_too_little_history_are_trained_on_but_not_scored():
    start = datetime(2025, 1, 1)
    data = Dataset(history(1, list(range(10)), start) + history(2, [99, 98], start))

    split = temporal_split(data, holdout=2, minimum_history=5)

    assert 2 not in split.test, "a user with two interactions cannot be scored meaningfully"
    assert {99, 98} <= {i.item for i in split.train.interactions}, "their data should still train"


def test_a_dataset_nobody_can_be_scored_on_says_so():
    data = Dataset(history(1, [1, 2], datetime(2025, 1, 1)))

    with pytest.raises(ValueError, match="no user had enough history"):
        temporal_split(data, holdout=2, minimum_history=5)


def test_holding_out_nothing_is_rejected():
    data = Dataset(history(1, list(range(10)), datetime(2025, 1, 1)))
    with pytest.raises(ValueError, match="at least one"):
        temporal_split(data, holdout=0)


def test_history_comes_back_in_time_order_however_the_file_was_written():
    start = datetime(2025, 1, 1)
    shuffled = list(reversed(history(1, [1, 2, 3], start)))
    data = Dataset(shuffled)

    assert [i.item for i in data.by_user()[1]] == [1, 2, 3]


def test_the_sample_dataset_is_sparse_enough_to_be_a_real_problem(dataset):
    """With fifty items and users who have seen forty, every model scores well
    and none of them is working. The catalogue has to be big enough that
    recommending is a choice."""
    density = len(dataset) / (len(dataset.users) * len(dataset.items))

    assert len(dataset.items) >= 250
    assert density < 0.2, f"the matrix is {density:.0%} full — too dense to evaluate on"


def test_the_sample_dataset_loads_with_titles_and_genres(dataset):
    assert len(dataset) == 17228
    assert len(dataset.users) == 500
    assert len(dataset.items) == 300
    assert all(dataset.name(i) for i in dataset.items)
    assert set(dataset.item_tags.values()) >= {"sci-fi", "comedy", "drama"}


def test_popularity_follows_a_long_tail(dataset):
    counts = sorted(dataset.popularity().values(), reverse=True)
    top_ten_share = sum(counts[:10]) / sum(counts)

    assert top_ten_share > 0.1, "a flat popularity curve makes the baseline meaningless"
