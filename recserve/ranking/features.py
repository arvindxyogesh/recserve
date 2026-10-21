"""Ranking features for (user, candidate item) pairs.

Torch-free so the serving process computes exactly the same features. Three
groups, all computed from interactions strictly before ``as_of_ts``:

  * user stats   (per user, precomputed and cached, e.g. in Redis)
  * item stats   (per item, precomputed table loaded in memory)
  * pair features (per request, from the user's recent positives and the
                   retrieval stage): retrieval score and rank, genre
                   affinity, genome tag similarity, release-year gap

Missing values are NaN; LightGBM learns a direction for them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from recserve.data.history import PAD
from recserve.data.item_features import ItemFeatures

DAY_S = 86_400

USER_STATS = [
    "u_n_ratings",
    "u_n_pos",
    "u_pos_rate",
    "u_mean_rating",
    "u_days_since_last",
    "u_days_active",
]
ITEM_STATS = [
    "i_n_ratings",
    "i_n_pos",
    "i_mean_rating",
    "i_pop_7d",
    "i_pop_90d",
    "i_days_since_first",
    "i_year",
    "i_has_genome",
]
PAIR = ["retrieval_score", "retrieval_rank", "genre_affinity", "genome_sim", "year_gap"]
FEATURES = USER_STATS + ITEM_STATS + PAIR


def user_stats(history: pd.DataFrame, n_users: int, as_of_ts: int) -> np.ndarray:
    g = history.groupby("user_idx").agg(
        n=("rating", "size"),
        pos=("label", "sum"),
        mean=("rating", "mean"),
        first=("ts", "min"),
        last=("ts", "max"),
    )
    out = np.full((n_users, len(USER_STATS)), np.nan, dtype=np.float32)
    idx = g.index.to_numpy()
    out[idx, 0] = g["n"]
    out[idx, 1] = g["pos"]
    out[idx, 2] = g["pos"] / g["n"]
    out[idx, 3] = g["mean"]
    out[idx, 4] = (as_of_ts - g["last"]) / DAY_S
    out[idx, 5] = (g["last"] - g["first"]) / DAY_S
    return out


def item_stats(
    history: pd.DataFrame, n_items: int, as_of_ts: int, feats: ItemFeatures
) -> np.ndarray:
    out = np.full((n_items, len(ITEM_STATS)), np.nan, dtype=np.float32)
    g = history.groupby("item_idx").agg(
        n=("rating", "size"), pos=("label", "sum"), mean=("rating", "mean"), first=("ts", "min")
    )
    idx = g.index.to_numpy()
    out[:, 0] = 0.0
    out[:, 1] = 0.0
    out[idx, 0] = g["n"]
    out[idx, 1] = g["pos"]
    out[idx, 2] = g["mean"]
    pos = history[history["label"] == 1]
    for col, days in ((3, 7), (4, 90)):
        recent = pos.loc[pos["ts"] >= as_of_ts - days * DAY_S, "item_idx"].to_numpy()
        out[:, col] = np.bincount(recent, minlength=n_items)
    out[idx, 5] = (as_of_ts - g["first"]) / DAY_S
    out[:, 6] = feats.year if feats.year is not None else np.nan
    out[:, 7] = feats.has_genome
    return out


class PairFeaturizer:
    """Builds user profiles from recent positives and scores candidates against them."""

    def __init__(self, feats: ItemFeatures) -> None:
        self.genres = feats.genres
        self.n_genres_item = np.maximum(feats.genres.sum(axis=1), 1.0)
        self.year = feats.year if feats.year is not None else np.full(feats.n_items, np.nan)
        self.has_genome = feats.has_genome.astype(bool)
        g = feats.genome if feats.genome is not None else np.zeros((feats.n_items, 1), np.float32)
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        self.genome_unit = np.divide(g, norms, out=np.zeros_like(g), where=norms > 0)

    def __call__(
        self,
        recent: np.ndarray,
        cand_items: np.ndarray,
        retrieval_score: np.ndarray,
        chunk: int = 256,
    ) -> np.ndarray:
        """recent: (U, L) padded positives; cand_items, retrieval_score: (U, N).

        Returns (U * N, len(PAIR)) float32 in row-major (user, candidate) order.
        Users are processed in chunks: the genome gather is (chunk, N, n_tags).
        """
        parts = [
            self._chunk(
                recent[s : s + chunk], cand_items[s : s + chunk], retrieval_score[s : s + chunk]
            )
            for s in range(0, len(recent), chunk)
        ]
        return np.concatenate(parts) if parts else np.zeros((0, len(PAIR)), np.float32)

    def _chunk(
        self, recent: np.ndarray, cand_items: np.ndarray, retrieval_score: np.ndarray
    ) -> np.ndarray:
        n_users, n_cand = cand_items.shape
        mask = recent != PAD
        safe = np.where(mask, recent, 0)
        n_hist = np.maximum(mask.sum(axis=1, keepdims=True), 1)

        genre_profile = (self.genres[safe] * mask[..., None]).sum(axis=1) / n_hist
        genre_aff = np.einsum("ug,ung->un", genre_profile, self.genres[cand_items])
        genre_aff /= self.n_genres_item[cand_items]

        gmask = mask & self.has_genome[safe]
        genome_profile = (self.genome_unit[safe] * gmask[..., None]).sum(axis=1)
        norms = np.linalg.norm(genome_profile, axis=1, keepdims=True)
        genome_profile = np.divide(
            genome_profile, norms, out=np.zeros_like(genome_profile), where=norms > 0
        )
        genome_sim = np.einsum("ut,unt->un", genome_profile, self.genome_unit[cand_items])
        genome_sim[~self.has_genome[cand_items] | (norms == 0)] = np.nan

        years = np.where(mask, self.year[safe], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN, expected
            mean_year = np.nanmean(years, axis=1)
        year_gap = np.abs(self.year[cand_items] - mean_year[:, None])

        rank = np.broadcast_to(np.arange(1, n_cand + 1, dtype=np.float32), (n_users, n_cand))
        cols = [retrieval_score, rank, genre_aff, genome_sim, year_gap]
        return np.stack([c.astype(np.float32).ravel() for c in cols], axis=1)


def assemble(user_rows: np.ndarray, item_rows: np.ndarray, pair: np.ndarray) -> np.ndarray:
    """Concatenate feature groups in FEATURES order; rows must already be aligned."""
    x = np.concatenate([user_rows, item_rows, pair], axis=1).astype(np.float32)
    assert x.shape[1] == len(FEATURES)
    return x
