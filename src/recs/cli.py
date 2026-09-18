"""recs: evaluate recommenders, or ask one for a page of recommendations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .data import load, temporal_split
from .evaluate import header, leaderboard
from .metrics import novelty
from .models import BPR, ItemKNN, MatrixFactorization, Popularity, default_models

MODELS = {
    "popularity": Popularity,
    "item-knn": ItemKNN,
    "mf": MatrixFactorization,
    "bpr": BPR,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recs", description=__doc__)
    parser.add_argument("--version", action="version", version=f"recs {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", type=Path, default=Path("data"))
        p.add_argument("--holdout", type=int, default=3, help="interactions held out per user")
        p.add_argument("-k", type=int, default=10, help="page size")

    describe = sub.add_parser("describe", help="what is in the dataset")
    common(describe)

    evaluate = sub.add_parser("evaluate", help="score every model on the same users")
    common(evaluate)
    evaluate.add_argument("--json", action="store_true")

    recommend = sub.add_parser("recommend", help="a page of recommendations for one user")
    common(recommend)
    recommend.add_argument("user", type=int)
    recommend.add_argument("--model", choices=sorted(MODELS), default="item-knn")
    recommend.add_argument("--explain", action="store_true",
                           help="show what the user already liked")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Wraps the real work so that piping into `head` — which closes
    the pipe early — ends quietly instead of printing a BrokenPipeError."""
    try:
        return _run(argv)
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as error:
        print(f"recs: {error}", file=sys.stderr)
        return 2


def _run(argv: list[str] | None) -> int:
    args = build_parser().parse_args(argv)
    data = load(args.data)

    if args.cmd == "describe":
        by_user = data.by_user()
        lengths = sorted(len(h) for h in by_user.values())
        counts = data.popularity()
        density = len(data) / (len(data.users) * len(data.items)) * 100
        print(f"{len(data):,} interactions   {len(data.users)} users   {len(data.items)} items")
        print(f"  matrix is {density:.1f}% full")
        print(f"  history per user: min {lengths[0]}  median {lengths[len(lengths) // 2]}  max {lengths[-1]}")
        print("  most interacted with:")
        for item, count in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
            print(f"    {data.name(item):<24} {data.item_tags.get(item, ''):<14} {count:>5}")
        tail = sum(1 for c in counts.values() if c <= 10)
        print(f"  {tail} of {len(counts)} items have 10 or fewer interactions")
        return 0

    split = temporal_split(data, holdout=args.holdout)

    if args.cmd == "evaluate":
        scores = leaderboard(default_models(), split, data, args.k)
        if args.json:
            print(json.dumps([{"model": s.model, "precision": round(s.precision, 4),
                               "recall": round(s.recall, 4), "map": round(s.map, 4),
                               "ndcg": round(s.ndcg, 4), "hit_rate": round(s.hit_rate, 4),
                               "coverage": round(s.coverage, 4)} for s in scores], indent=2))
            return 0
        print(f"{len(split.evaluable_users)} users, last {args.holdout} interactions held out\n")
        print(header(args.k))
        for score in scores:
            print(score.row())
        return 0

    model = MODELS[args.model]()
    model.fit(split.train)
    page = model.recommend(args.user, args.k)
    if not page:
        print(f"recs: nothing to recommend for user {args.user} — no history in the training half",
              file=sys.stderr)
        return 1

    if args.explain:
        history = [i for i in split.train.interactions if i.user == args.user]
        history.sort(key=lambda i: -i.rating)
        print(f"user {args.user} rated highest:")
        for interaction in history[:5]:
            print(f"  {interaction.rating:>4.1f}  {data.name(interaction.item):<24}"
                  f"{data.item_tags.get(interaction.item, '')}")
        print()

    counts = data.popularity()
    print(f"{model.name} recommends for user {args.user}:")
    for rank, item in enumerate(page, start=1):
        print(f"  {rank:>2}. {data.name(item):<24}{data.item_tags.get(item, ''):<14}"
              f"{counts.get(item, 0):>5} interactions")
    print(f"\n  novelty {novelty([page], counts, len(data), args.k):.2f} bits")
    return 0
