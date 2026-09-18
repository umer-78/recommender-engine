import pytest

from recs import (
    BPR,
    ItemKNN,
    MatrixFactorization,
    Popularity,
    RandomRecommender,
    evaluate,
)


@pytest.fixture(scope="module")
def fitted(split):
    """Every model fitted once. Training BPR per test would dominate the run."""
    models = {
        "random": RandomRecommender(),
        "popularity": Popularity(),
        "item-knn": ItemKNN(neighbours=50, shrinkage=20),
        "mf": MatrixFactorization(factors=16, epochs=12),
        "bpr": BPR(factors=32, epochs=25),
    }
    for model in models.values():
        model.fit(split.train)
    return models


def test_no_model_ever_recommends_something_the_user_has_already_seen(fitted, split):
    """The oldest bug in recommendation.

    The best predictor of what someone watched is what they watched, so leaving
    seen items in makes every metric look wonderful and ships a product that
    recommends the film you finished last night.
    """
    seen = split.train.by_user()

    for name, model in fitted.items():
        for user in split.evaluable_users[:40]:
            already = {i.item for i in seen[user]}
            page = model.recommend(user, 10)
            assert not (set(page) & already), f"{name} recommended something user {user} has seen"


def test_pages_are_the_size_asked_for(fitted, split):
    user = split.evaluable_users[0]
    for name, model in fitted.items():
        assert len(model.recommend(user, 5)) == 5, name
        assert len(model.recommend(user, 20)) == 20, name


def test_popularity_shows_everyone_the_same_thing(fitted, split):
    model = fitted["popularity"]
    a, b = split.evaluable_users[0], split.evaluable_users[1]

    # Only the items each has already seen differ between the two pages.
    assert len(set(model.recommend(a, 20)) & set(model.recommend(b, 20))) >= 10


def test_item_knn_stays_near_what_the_user_already_likes(dataset, fitted, split):
    model = fitted["item-knn"]
    histories = split.train.by_user()

    matched = 0
    for user in split.evaluable_users[:50]:
        liked = [i for i in histories[user] if i.rating >= 4]
        if not liked:
            continue
        favourite_genres = {dataset.item_tags[i.item] for i in liked}
        page = model.recommend(user, 10)
        overlap = sum(1 for item in page if dataset.item_tags[item] in favourite_genres)
        matched += overlap >= 4

    assert matched > 30, "recommendations should mostly sit in genres the user rates highly"


def test_an_unknown_user_gets_an_empty_page_rather_than_an_error(fitted):
    for name in ("item-knn", "mf", "bpr"):
        assert fitted[name].recommend(999_999, 10) == [], name


def test_shrinkage_suppresses_similarity_built_on_few_shared_users(split):
    """Three co-occurrences can produce a cosine of 1.0. Without shrinkage those
    accidents dominate the neighbourhood of every obscure item."""
    raw = ItemKNN(neighbours=50, shrinkage=0).fit(split.train)
    shrunk = ItemKNN(neighbours=50, shrinkage=50).fit(split.train)

    item = next(iter(sorted(split.train.items)))
    top_raw = raw._similar[item][0][1]
    top_shrunk = shrunk._similar[item][0][1]

    assert top_shrunk < top_raw


def test_matrix_factorisation_reduces_its_training_error(split):
    model = MatrixFactorization(factors=16, epochs=12).fit(split.train)
    assert model.losses[-1] < model.losses[0]


def test_bpr_reduces_its_ranking_loss(split):
    model = BPR(factors=16, epochs=15).fit(split.train)
    assert model.losses[-1] < model.losses[0] / 2


def test_rating_prediction_does_not_rank(fitted, split, dataset):
    """The finding this project exists to show.

    MatrixFactorization minimises squared error on observed ratings and is then
    used to order items. Its highest predicted scores go to items a handful of
    generous users rated well, which nobody else wants — and ranked that way it
    lands *below a random shuffle*.
    """
    mf = evaluate(fitted["mf"], split, dataset, k=10)
    shuffle = evaluate(fitted["random"], split, dataset, k=10)

    assert mf.ndcg < shuffle.ndcg


def test_the_same_factorisation_trained_on_a_ranking_loss_works(fitted, split, dataset):
    """BPR is the same architecture with the ratings thrown away and a pairwise
    order loss in their place. That single change is the difference between worse
    than random and competitive."""
    mf = evaluate(fitted["mf"], split, dataset, k=10)
    bpr = evaluate(fitted["bpr"], split, dataset, k=10)

    assert bpr.ndcg > mf.ndcg * 5


def test_personalisation_beats_popularity_beats_random(fitted, split, dataset):
    scores = {name: evaluate(model, split, dataset, k=10).ndcg for name, model in fitted.items()}

    assert scores["item-knn"] > scores["popularity"] > scores["random"]
    assert scores["bpr"] > scores["popularity"]


def test_popularity_reaches_almost_none_of_the_catalogue(fitted, split, dataset):
    """Accuracy is not the only thing that matters: a recommender that shows the
    same few dozen items to everybody can score respectably and still be useless."""
    popularity = evaluate(fitted["popularity"], split, dataset, k=10)
    bpr = evaluate(fitted["bpr"], split, dataset, k=10)

    assert popularity.coverage < 0.2
    assert bpr.coverage > 0.8
