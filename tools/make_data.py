"""Writes the sample catalogue and interaction log in data/.

Seeded, so the figures in the README are reproducible and CI can check the files
have not drifted. The generator gives users genre preferences and items a
power-law popularity, because a recommender evaluated on uniform random data
learns nothing and scores like it.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 20260918
DATA = Path(__file__).resolve().parent.parent / "data"

GENRES = ["sci-fi", "thriller", "comedy", "drama", "documentary", "animation"]

# Titles are assembled from word lists rather than written out: the catalogue has
# to be big enough that recommending is a real problem. With fifty items and users
# who have seen forty of them, every model scores well and none of them is working.
SEEDS = {
    "sci-fi": (["Orbital", "Ion", "Vacuum", "Kepler", "Titan", "Quiet", "Second", "Deep",
                "Silent", "Outer"],
               ["Drift", "Harvest", "State", "Signal", "Freight", "Nebula", "Sun", "Field",
                "Transit", "Horizon"]),
    "thriller": (["Cold", "Night", "Paper", "Safe", "Blind", "Dead", "Last", "Silent",
                  "Broken", "Quiet"],
                 ["Open", "Audit", "Trail", "House", "Copy", "Drop", "Courier", "Ledger",
                  "Witness", "Hour"]),
    "comedy": (["Office", "Group", "Small", "Refund", "Wedding", "Spare", "Free", "Late",
                "Open", "Second"],
               ["Hours", "Chat", "Claims", "Policy", "Tax", "Room", "Trial", "Shift",
                "Mic", "Helping"]),
    "drama": (["Harbour", "Salt", "Winter", "Low", "Long", "Grey", "Still", "Near",
               "Old", "Far"],
              ["Light", "Flats", "Term", "Tide", "Field", "Water", "Harvest", "Country",
               "Ground", "Shore"]),
    "documentary": (["Deep", "Seed", "Night", "Open", "Hard", "Clean", "Raw", "Slow",
                     "Whole", "Last"],
                    ["Water", "Vault", "Shift", "Grid", "Copper", "Glass", "Commute",
                     "Harvest", "Coast", "Mile"]),
    "animation": (["Paper", "Cloud", "Pocket", "Marmalade", "Bell", "Tin", "Glass", "Honey",
                   "Copper", "Feather"],
                  ["Moon", "Line", "Watch", "Bay", "Whistle", "Town", "Garden", "Lantern",
                   "Kite", "Road"]),
}


def build() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)

    items = []
    item_id = 1
    for genre in GENRES:
        firsts, seconds = SEEDS[genre]
        titles = sorted({f"{a} {b}" for a in firsts for b in seconds})
        rng.shuffle(titles)
        for title in titles[:50]:
            items.append({"item_id": item_id, "title": title, "genre": genre,
                          "year": rng.randint(2008, 2025)})
            item_id += 1

    # A power-law appeal: a few items everyone runs into, a long tail nobody does.
    appeal = {item["item_id"]: rng.paretovariate(1.3) for item in items}
    by_genre: dict[str, list[int]] = {g: [] for g in GENRES}
    for item in items:
        by_genre[item["genre"]].append(item["item_id"])

    interactions = []
    start = datetime(2025, 1, 1, 9, 0)
    for user in range(1, 501):
        # Each user has one or two genres they mostly stay in.
        favourites = rng.sample(GENRES, rng.choice([1, 1, 2]))
        generosity = rng.gauss(0, 0.5)          # some users rate everything highly
        count = rng.randint(10, 60)
        when = start + timedelta(days=rng.randint(0, 200), hours=rng.randint(0, 12))

        chosen: set[int] = set()
        while len(chosen) < count:
            genre = rng.choice(favourites) if rng.random() < 0.75 else rng.choice(GENRES)
            pool = by_genre[genre]
            weights = [appeal[i] for i in pool]
            item = rng.choices(pool, weights=weights)[0]
            if item in chosen:
                continue
            chosen.add(item)

            base = 3.6 + (0.6 if any(i["item_id"] == item and i["genre"] in favourites
                                     for i in items) else -0.4)
            rating = min(5.0, max(1.0, round(base + generosity + rng.gauss(0, 0.6), 1)))
            when += timedelta(hours=rng.randint(4, 96))
            interactions.append({"user_id": user, "item_id": item, "rating": rating,
                                 "timestamp": when.isoformat(timespec="seconds")})

    interactions.sort(key=lambda r: (r["timestamp"], r["user_id"]))
    return items, interactions


def write(name: str, rows: list[dict]) -> None:
    path = DATA / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    items, interactions = build()
    write("items.csv", items)
    write("interactions.csv", interactions)


if __name__ == "__main__":
    main()
