"""Offline evaluation harness shared by every retrieval and ranking model.

Protocol for an evaluation period P (val or test):
  * history = every interaction strictly before P (train for val,
    train + val for test). Models are fit on the history only.
  * ground truth = each user's positives inside P.
  * evaluated users = users with >= 1 positive in P and >= 1 positive in
    history ("warm"). Cold users are evaluated separately, only by models
    that can serve them.
  * items the user rated in history (positive or not) are masked out. In
    MovieLens a user rates a movie at most once, so a held-out item is never
    masked by mistake.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recserve.eval.metrics import hits_matrix, metrics_from_hits


@dataclass
class EvalContext:
    """Everything a model may use when fit for one evaluation period."""

    name: str  # "val" or "test"
    history: pd.DataFrame  # all interactions before the period
    target: pd.DataFrame  # interactions inside the period
    n_users: int
    n_items: int
    start_ts: int  # first timestamp of the period (= history cutoff)

    def __post_init__(self) -> None:
        self.history_pos = to_csr(
            self.history[self.history["label"] == 1], self.n_users, self.n_items
        )
        self.history_all = to_csr(self.history, self.n_users, self.n_items)
        self.target_pos = to_csr(self.target[self.target["label"] == 1], self.n_users, self.n_items)

    def users(self, kind: str = "warm") -> np.ndarray:
        has_target = np.diff(self.target_pos.indptr) > 0
        has_history = np.diff(self.history_pos.indptr) > 0
        if kind == "warm":
            mask = has_target & has_history
        elif kind == "cold":
            mask = has_target & ~has_history
        else:
            raise ValueError(f"unknown user kind {kind!r}")
        return np.flatnonzero(mask).astype(np.int32)


class Recommender(Protocol):
    name: str

    def fit(self, ctx: EvalContext) -> None: ...

    def score(self, users: np.ndarray, ctx: EvalContext) -> np.ndarray:
        """Dense (len(users), n_items) float32 scores; higher is better."""
        ...


def to_csr(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    data = np.ones(len(df), dtype=np.float32)
    m = sp.csr_matrix(
        (data, (df["user_idx"].to_numpy(), df["item_idx"].to_numpy())), shape=(n_users, n_items)
    )
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, sorted by descending score."""
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def evaluate(
    model: Recommender,
    ctx: EvalContext,
    retrieval_ks: list[int],
    ranking_k: int,
    batch_size: int = 1024,
    users: np.ndarray | None = None,
    max_users: int | None = None,
    rng: np.random.Generator | None = None,
    fit: bool = True,
) -> dict[str, Any]:
    if users is None:
        users = ctx.users("warm")
    if max_users is not None and len(users) > max_users:
        rng = rng or np.random.default_rng(0)
        users = np.sort(rng.choice(users, size=max_users, replace=False))
    k_max = max(*retrieval_ks, ranking_k)

    fit_s = 0.0
    if fit:
        t0 = time.perf_counter()
        model.fit(ctx)
        fit_s = time.perf_counter() - t0

    all_hits = []
    score_s = 0.0
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        t0 = time.perf_counter()
        scores = model.score(batch, ctx)
        seen = ctx.history_all[batch]
        scores[seen.nonzero()] = -np.inf
        ranked = top_k(scores, k_max)
        score_s += time.perf_counter() - t0
        all_hits.append(hits_matrix(ranked, batch, ctx.target_pos))

    hits = np.concatenate(all_hits) if all_hits else np.zeros((0, k_max), dtype=bool)
    n_rel = np.diff(ctx.target_pos.indptr)[users]
    metrics = metrics_from_hits(hits, n_rel, retrieval_ks, ranking_k) if len(users) else {}
    return {
        "model": model.name,
        "period": ctx.name,
        "n_users": int(len(users)),
        "mean_relevant_per_user": float(n_rel.mean()) if len(users) else 0.0,
        "metrics": metrics,
        "fit_s": round(fit_s, 3),
        "score_ms_per_user": round(1000 * score_s / max(len(users), 1), 4),
    }
