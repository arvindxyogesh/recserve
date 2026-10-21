"""Item content features aligned to item_idx: genres, release-year bucket, genome tags."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
YEAR_MIN, YEAR_MAX, YEAR_STEP = 1900, 2029, 5


@dataclass
class ItemFeatures:
    genres: np.ndarray  # (n_items, n_genres) float32 multi-hot
    genre_names: list[str]
    year_bucket: np.ndarray  # (n_items,) int64, 0 = unknown year
    n_year_buckets: int
    genome: np.ndarray | None  # (n_items, n_tags) float32, zeros where missing
    has_genome: np.ndarray  # (n_items,) float32

    @property
    def n_items(self) -> int:
        return len(self.year_bucket)


def year_bucket(title: str) -> int:
    match = YEAR_RE.search(title.strip())
    if not match:
        return 0
    year = min(max(int(match.group(1)), YEAR_MIN), YEAR_MAX)
    return 1 + (year - YEAR_MIN) // YEAR_STEP


def build_item_features(
    processed_dir: str | Path, item_map: pd.DataFrame, use_genome: bool = True
) -> ItemFeatures:
    processed_dir = Path(processed_dir)
    movies = pd.read_parquet(processed_dir / "movies.parquet")
    movies = item_map.merge(movies, on="movieId", how="left").sort_values("item_idx")
    n_items = len(item_map)

    genre_lists = movies["genres"].fillna("").str.split("|")
    names = sorted({g for gs in genre_lists for g in gs if g})
    col = {g: j for j, g in enumerate(names)}
    genres = np.zeros((n_items, len(names)), dtype=np.float32)
    for i, gs in enumerate(genre_lists):
        for g in gs:
            if g:
                genres[i, col[g]] = 1.0

    years = np.array([year_bucket(t) for t in movies["title"].fillna("")], dtype=np.int64)
    n_buckets = 2 + (YEAR_MAX - YEAR_MIN) // YEAR_STEP

    has_genome = np.zeros(n_items, dtype=np.float32)
    genome = None
    if use_genome:
        scores = pd.read_parquet(processed_dir / "genome_scores.parquet")
        scores = scores.merge(item_map, on="movieId")
        tag_ids = np.sort(scores["tagId"].unique())
        genome = np.zeros((n_items, len(tag_ids)), dtype=np.float32)
        genome[scores["item_idx"].to_numpy(), np.searchsorted(tag_ids, scores["tagId"])] = scores[
            "relevance"
        ].to_numpy()
        has_genome[scores["item_idx"].unique()] = 1.0

    return ItemFeatures(genres, names, years, n_buckets, genome, has_genome)
