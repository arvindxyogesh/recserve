"""Top-K recommendation metrics with binary relevance.

Two implementations are kept on purpose:
  * scalar per-user functions (``recall_at_k`` etc.) are the readable
    reference definitions used in unit tests;
  * ``hits_matrix`` + ``metrics_from_hits`` compute the same quantities
    vectorised over many users and are what the evaluation harness uses.
The unit tests check that both agree.

Definitions (ranked list r, relevant set R, 1-based rank i):
  Recall@K   = |r[:K] & R| / |R|
  HitRate@K  = 1 if |r[:K] & R| > 0 else 0
  MRR@K      = 1 / (rank of first relevant item in r[:K]), 0 if none
  NDCG@K     = DCG@K / IDCG@K, DCG@K = sum_{i<=K, r_i in R} 1 / log2(i + 1),
               IDCG@K = sum_{i <= min(K, |R|)} 1 / log2(i + 1)
All metrics are averaged over users with |R| > 0.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import scipy.sparse as sp


def _check(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def recall_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    _check(k)
    rel = set(relevant)
    if not rel:
        raise ValueError("relevant set is empty")
    return len(set(ranked[:k]) & rel) / len(rel)


def hit_rate_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    _check(k)
    rel = set(relevant)
    return float(any(item in rel for item in ranked[:k]))


def reciprocal_rank_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    _check(k)
    rel = set(relevant)
    for i, item in enumerate(ranked[:k], start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    _check(k)
    rel = set(relevant)
    if not rel:
        raise ValueError("relevant set is empty")
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(rel))))
    return dcg / idcg


def hits_matrix(topk: np.ndarray, users: np.ndarray, relevant: sp.csr_matrix) -> np.ndarray:
    """Boolean (n_users, K) matrix: hits[u, j] is True if topk[u, j] is relevant to users[u].

    Membership is tested by encoding (user, item) pairs as int64 keys and using a
    sorted search, which is much faster than per-user Python sets.
    """
    n_items = relevant.shape[1]
    coo = relevant.tocoo()
    rel_keys = np.sort(coo.row.astype(np.int64) * n_items + coo.col)
    keys = users.astype(np.int64)[:, None] * n_items + topk.astype(np.int64)
    pos = np.searchsorted(rel_keys, keys.ravel())
    pos = np.minimum(pos, len(rel_keys) - 1)
    return (rel_keys[pos] == keys.ravel()).reshape(topk.shape)


def metrics_from_hits(
    hits: np.ndarray, n_relevant: np.ndarray, retrieval_ks: Sequence[int], ranking_k: int
) -> dict[str, float]:
    """Mean metrics from a hits matrix (users x K) and each user's number of relevant items."""
    n_relevant = np.asarray(n_relevant, dtype=np.float64)
    if hits.shape[0] != len(n_relevant):
        raise ValueError("hits and n_relevant disagree on number of users")
    if np.any(n_relevant <= 0):
        raise ValueError("every evaluated user needs at least one relevant item")
    max_k = hits.shape[1]
    for k in (*retrieval_ks, ranking_k):
        if k > max_k:
            raise ValueError(f"k={k} larger than the ranked list length {max_k}")

    out: dict[str, float] = {}
    cum = np.cumsum(hits, axis=1)
    for k in retrieval_ks:
        out[f"recall@{k}"] = float(np.mean(cum[:, k - 1] / n_relevant))

    k = ranking_k
    per_user = per_user_ranking(hits, n_relevant, k)
    out[f"recall@{k}"] = float(np.mean(cum[:, k - 1] / n_relevant))
    for name, values in per_user.items():
        out[f"{name}@{k}"] = float(np.mean(values))
    return out


def per_user_ranking(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Per-user NDCG@k, reciprocal rank@k and hit@k (for paired tests between models)."""
    n_relevant = np.asarray(n_relevant, dtype=np.float64)
    top = hits[:, :k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = top @ discounts
    ideal_len = np.minimum(n_relevant, k).astype(int)
    idcg = np.concatenate([[0.0], np.cumsum(discounts)])[ideal_len]
    first = np.argmax(top, axis=1)
    return {
        "ndcg": dcg / idcg,
        "mrr": np.where(top.any(axis=1), 1.0 / (first + 1), 0.0),
        "hitrate": top.any(axis=1).astype(np.float64),
    }


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """CI for mean(a) - mean(b) and relative lift, resampling users (paired)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diff = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lift = diff / b[idx].mean(axis=1)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "diff": float(a.mean() - b.mean()),
        "diff_ci": [float(np.percentile(diff, lo)), float(np.percentile(diff, hi))],
        "lift": float(a.mean() / b.mean() - 1),
        "lift_ci": [float(np.percentile(lift, lo)), float(np.percentile(lift, hi))],
    }
