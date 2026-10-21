"""Non-learned retrieval baselines: random, popularity and item-item co-occurrence."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recserve.eval.harness import EvalContext, to_csr
from recserve.utils import log

DAY_S = 86_400


class RandomRecommender:
    """Uniform random scores: the floor every model must beat."""

    def __init__(self, seed: int = 0) -> None:
        self.name = "random"
        self.seed = seed

    def fit(self, ctx: EvalContext) -> None:
        self.n_items = ctx.n_items
        self.rng = np.random.default_rng(self.seed)

    def score(self, users: np.ndarray, ctx: EvalContext) -> np.ndarray:
        return self.rng.random((len(users), self.n_items), dtype=np.float32)


class PopularityRecommender:
    """Rank items by number of positives, optionally only in the last ``window_days``.

    The recent variant matters with a temporal split: the evaluation period is
    after the history, and what is popular now differs from all-time favourites.
    """

    def __init__(self, window_days: int | None = None) -> None:
        self.window_days = window_days
        self.name = "popularity" if window_days is None else f"popularity_{window_days}d"

    def fit(self, ctx: EvalContext) -> None:
        pos = ctx.history[ctx.history["label"] == 1]
        if self.window_days is not None:
            pos = pos[pos["ts"] >= ctx.start_ts - self.window_days * DAY_S]
        counts = np.bincount(pos["item_idx"].to_numpy(), minlength=ctx.n_items)
        self.item_scores = counts.astype(np.float32)

    def score(self, users: np.ndarray, ctx: EvalContext) -> np.ndarray:
        return np.tile(self.item_scores, (len(users), 1))


class CooccurrenceRecommender:
    """Item-item collaborative filtering on co-occurrence of positives.

    sim(i, j) = c_ij / (sqrt(n_i) * sqrt(n_j) + shrinkage), the shrunk cosine
    between the binary user vectors of items i and j. Only the ``top_neighbors``
    most similar items are kept per item. A user's score for item j is the sum
    of sim(i, j) over the items i in their positive history.

    Items with zero score are backfilled in popularity order (their score is
    set into [-1, 0)), so every user gets a full ranked list.
    """

    def __init__(
        self,
        max_history: int = 200,
        top_neighbors: int = 200,
        shrinkage: float = 10.0,
        min_item_positives: int = 5,
        chunk_size: int = 1024,
    ) -> None:
        self.name = "cooccurrence"
        self.max_history = max_history
        self.top_neighbors = top_neighbors
        self.shrinkage = shrinkage
        self.min_item_positives = min_item_positives
        self.chunk_size = chunk_size

    def fit(self, ctx: EvalContext) -> None:
        pos = ctx.history[ctx.history["label"] == 1]
        # Cap each user's contribution to their most recent positives: bounds the
        # O(sum of history^2) cost and stops a few heavy raters dominating counts.
        recent = (
            pos.sort_values("ts", ascending=False, kind="stable")
            .groupby("user_idx", sort=False)
            .head(self.max_history)
        )
        x = to_csr(recent, ctx.n_users, ctx.n_items)
        self.sim = self._neighbours(x)

        pop = np.bincount(pos["item_idx"].to_numpy(), minlength=ctx.n_items).astype(np.float64)
        ranks = np.argsort(np.argsort(pop, kind="stable"), kind="stable")
        self.backfill = (ranks / ctx.n_items - 1.0).astype(np.float32)  # in [-1, 0)

    def _neighbours(self, x: sp.csr_matrix) -> sp.csr_matrix:
        n_items = x.shape[1]
        n = np.asarray(x.sum(axis=0)).ravel().astype(np.float32)
        eligible = n >= self.min_item_positives
        rows_all = np.flatnonzero(eligible)
        root_n = np.sqrt(n)
        xc = x.tocsc()
        k = min(self.top_neighbors, int(eligible.sum()) - 1)

        rows, cols, vals = [], [], []
        for start in range(0, len(rows_all), self.chunk_size):
            chunk = rows_all[start : start + self.chunk_size]
            co = (xc[:, chunk].T.tocsr() @ x).toarray()  # (chunk, n_items) co-counts
            with np.errstate(divide="ignore", invalid="ignore"):  # 0/0 only for n_j = 0,
                sim = co / (root_n[chunk, None] * root_n[None, :] + self.shrinkage)
            sim[:, ~eligible] = 0.0  # ... and those columns are zeroed here
            sim[np.arange(len(chunk)), chunk] = 0.0
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            top_vals = np.take_along_axis(sim, top, axis=1)
            keep = top_vals > 0
            rows.append(np.repeat(chunk, k)[keep.ravel()])
            cols.append(top[keep])
            vals.append(top_vals[keep].astype(np.float32))
        sim_matrix = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_items, n_items),
        )
        log.info("co-occurrence: %d items with neighbours, %d edges", len(rows_all), sim_matrix.nnz)
        return sim_matrix

    def score(self, users: np.ndarray, ctx: EvalContext) -> np.ndarray:
        scores = (ctx.history_pos[users] @ self.sim).toarray().astype(np.float32)
        zero = scores <= 0
        scores[zero] = np.broadcast_to(self.backfill, scores.shape)[zero]
        return scores
