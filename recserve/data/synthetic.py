"""Generate a tiny MovieLens-shaped dataset for tests and the CI smoke test.

Writes the same Parquet files and schema as recserve.data.download, so every
downstream script runs unchanged. Users have latent genre preferences, which
gives the data enough structure for personalised models to beat popularity.

Usage::

    python -m recserve.data.synthetic --config configs/smoke.yaml
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recserve.config import config_from_cli
from recserve.utils import log, setup_logging

GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Animation"]


def make_synthetic(
    n_users: int = 400, n_items: int = 300, ratings_per_user: int = 40, n_tags: int = 20,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    item_genre = rng.integers(0, len(GENRES), size=n_items)
    item_pop = rng.zipf(1.6, size=n_items).clip(max=50).astype(float)
    user_genre = rng.dirichlet(np.full(len(GENRES), 0.3), size=n_users)
    movie_ids = np.sort(rng.choice(np.arange(1, n_items * 3), size=n_items, replace=False))

    rows = []
    t0 = 1_400_000_000
    for u in range(n_users):
        affinity = user_genre[u, item_genre]
        p = affinity * item_pop
        p /= p.sum()
        n = min(ratings_per_user + int(rng.integers(-10, 10)), n_items)
        items = rng.choice(n_items, size=n, replace=False, p=p)
        start = t0 + int(rng.integers(0, 300 * 86_400))
        ts = start + np.sort(rng.integers(0, 200 * 86_400, size=n))
        rating = np.clip(np.round((1.5 + 4 * affinity[items] + rng.normal(0, 0.8, n)) * 2) / 2,
                         0.5, 5.0)
        rows.append(pd.DataFrame({"userId": u + 1, "movieId": movie_ids[items],
                                  "rating": rating, "timestamp": ts}))
    ratings = pd.concat(rows, ignore_index=True).astype(
        {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"})

    movies = pd.DataFrame({
        "movieId": movie_ids.astype("int32"),
        "title": [f"Movie {m} (2000)" for m in movie_ids],
        "genres": [GENRES[g] for g in item_genre],
    })
    tags = ratings.sample(frac=0.05, random_state=seed)[["userId", "movieId", "timestamp"]]
    tags = tags.assign(tag=[f"tag{t}" for t in rng.integers(0, n_tags, len(tags))])
    genome_tags = pd.DataFrame({"tagId": np.arange(1, n_tags + 1, dtype="int16"),
                                "tag": [f"tag{t}" for t in range(n_tags)]})
    genome_scores = pd.DataFrame({
        "movieId": np.repeat(movie_ids, n_tags).astype("int32"),
        "tagId": np.tile(genome_tags["tagId"].to_numpy(), n_items),
        "relevance": rng.random(n_items * n_tags).astype("float32"),
    })
    links = pd.DataFrame({"movieId": movie_ids.astype("int32"),
                          "imdbId": [f"{m:07d}" for m in movie_ids], "tmdbId": movie_ids * 1.0})
    return {"ratings": ratings, "movies": movies, "tags": tags[["userId", "movieId", "tag",
                                                                "timestamp"]],
            "links": links, "genome_scores": genome_scores, "genome_tags": genome_tags}


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    out = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name, df in make_synthetic(seed=cfg["seed"], **cfg["data"].get("synthetic", {})).items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        log.info("%s: %d rows", name, len(df))


if __name__ == "__main__":
    main()
