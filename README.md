# recs

[![CI](https://github.com/umer-78/recommender-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/umer-78/recommender-engine/actions/workflows/ci.yml)

**Live demo:** https://umer-78.github.io/recommender-engine/

A recommender engine in Python with no dependencies: popularity and random
baselines, item-item collaborative filtering, matrix factorisation, and BPR —
scored on a temporal split, on ranking metrics, with the baselines left in.

```
$ recs evaluate -k 10
500 users, last 3 interactions held out

model                       P@10      R@10    MAP@10   NDCG@10      hit  coverage
item-knn                  0.0848    0.2827    0.1205    0.2059    0.630     0.553
bpr                       0.0762    0.2540    0.1088    0.1878    0.592     0.970
popularity                0.0458    0.1527    0.0640    0.1135    0.380     0.107
random                    0.0122    0.0407    0.0116    0.0256    0.122     1.000
matrix factorisation      0.0070    0.0233    0.0060    0.0138    0.068     0.163
```

Read that table from the bottom. **Matrix factorisation ranks worse than a random
shuffle.** That is not a bug in the implementation — it is what happens when you
train a model to predict ratings and then use it to order things.

- **32 tests**, Python 3.10–3.12, no runtime dependencies
- 17,228 interactions, 500 users, 300 items, generated from a fixed seed

## Quick start

```bash
git clone https://github.com/umer-78/recommender-engine.git
cd recommender-engine
pip install -e ".[dev]"
pytest -q                       # 32 tests

recs describe
recs evaluate -k 10
recs recommend 42 --model item-knn --explain
```

## The thing worth reading

**Rating prediction is not ranking.** `MatrixFactorization` minimises squared
error on observed ratings — the classic Netflix-prize objective — and scores well
at exactly that. Rank items by its predicted rating and it collapses: the highest
predictions go to items a handful of generous users rated 5, which almost nobody
else wants. On this data it reaches an NDCG@10 of **0.0138**, against a random
shuffle's **0.0256**.

**BPR is the same architecture with the ratings thrown away.** For each thing a
user interacted with, sample something they did not, and push the first above the
second — the gradient of `-log sigmoid(score_positive − score_negative)`. Nothing
is optimised except the order, which is the only thing the metrics measure. Same
factors, same regularisation, same data: **0.0138 → 0.1878**, a factor of
thirteen.

Both numbers are pinned in tests (`test_rating_prediction_does_not_rank`,
`test_the_same_factorisation_trained_on_a_ranking_loss_works`), so the finding
cannot quietly stop being true.

## Three more decisions that decide whether the numbers mean anything

**The split is by time, per user — never at random.** A random split lets a model
train on what someone did in December and be tested on what they did in March.
That is information no deployed recommender has. Scores from a random split are
routinely double what the same model achieves in production, and this is the most
common reason a recommender looks good in a notebook and lands flat. A test walks
the folds and asserts no training interaction is later than a held-out one.

**Seen items are excluded from every page.** The best predictor of what someone
watched is what they watched. Leave seen items in and every metric looks
wonderful while the product recommends the film you finished last night. One test
checks every model, for forty users, that no page overlaps their history.

**Coverage is reported next to accuracy.** Popularity reaches **10.7%** of the
catalogue: it shows nearly everybody the same thirty-odd items. BPR reaches
**97%**. Item-KNN wins on accuracy but shows only **55%**. A recommender judged on
accuracy alone will happily converge on a bestseller shelf, and the number that
catches it has to be in the same table.

## Models

| Model | Learns from | Optimises |
| --- | --- | --- |
| `RandomRecommender` | nothing | nothing — the floor |
| `Popularity` | interaction counts | nothing; a strong baseline most comparisons omit |
| `ItemKNN` | co-occurrence | cosine similarity, shrunk toward zero when two items share few users |
| `MatrixFactorization` | ratings | squared error — good at rating prediction, bad at ranking |
| `BPR` | which items were touched | pairwise order |

Item-KNN's shrinkage matters more than it looks: three co-occurrences can produce
a cosine of 1.0, and without the `count / (count + shrinkage)` factor those
accidents dominate the neighbourhood of every obscure item. A test compares a
shrunk model against an unshrunk one and requires the top similarity to fall.

## A page of recommendations

```
$ recs recommend 42 --model item-knn --explain
user 42 rated highest:
   4.7  Slow Glass              documentary
   4.5  Raw Commute             documentary
   4.4  Spare Chat              comedy
   4.1  Hard Coast              documentary
   4.0  Whole Harvest           documentary

item-knn recommends for user 42:
   1. Broken Witness          thriller        357 interactions
   2. Hard Harvest            documentary     158 interactions
   3. Blind Audit             thriller        215 interactions
   4. Slow Shift              documentary     103 interactions
   5. Second State            sci-fi          262 interactions
   6. Bell Garden             animation       239 interactions
   7. Orbital Harvest         sci-fi          227 interactions
   8. Hard Shift              documentary      88 interactions
   9. Orbital Horizon         sci-fi          201 interactions
  10. Office Room             comedy          118 interactions

  novelty 6.58 bits
```

## The data

```
$ recs describe
17,228 interactions   500 users   300 items
  matrix is 11.5% full
  history per user: min 10  median 35  max 60
  most interacted with:
    Spare Chat               comedy           397
    Broken Witness           thriller         357
    Whole Harvest            documentary      277
    Second State             sci-fi           262
    Salt Term                drama            259
```

Generated by `tools/make_data.py` from a fixed seed — titles assembled from word
lists, users given one or two preferred genres, items given a power-law popularity. The
catalogue size is deliberate: with fifty items and users who have seen forty of
them, every model scores well and none of them is working. A test asserts the
matrix stays under 20% full for exactly that reason.

CI regenerates the files and fails if they differ from what is committed.

## As a library

```python
from recs import load, temporal_split, BPR, evaluate

data = load("data")
split = temporal_split(data, holdout=3)

model = BPR(factors=32, epochs=40).fit(split.train)
model.recommend(user=42, k=10)          # [117, 8, 240, ...]

score = evaluate(model, split, data, k=10)
score.ndcg, score.coverage              # (0.1878, 0.97)
```

## Layout

```
src/recs/metrics.py     precision, recall, MAP, NDCG, hit rate, coverage, novelty
src/recs/models.py      random, popularity, item-KNN, MF, BPR
src/recs/data.py        loading and the temporal split
src/recs/evaluate.py    the leaderboard
src/recs/cli.py         the `recs` command
data/                   300 items, 17k interactions, from a fixed seed
tools/make_data.py      the generator; CI fails if the output drifts
tests/                  32 tests
```

## Not included

Implicit-feedback ALS, neural recommenders, session-based models, side features
for cold start, online A/B machinery. What is here is the part that has to be
right before any of that helps: a split that does not leak, metrics that measure
ranking, and baselines you are not allowed to hide.

## Licence

MIT — see [LICENSE](LICENSE).
