import math

import numpy as np
import pytest
import scipy.sparse as sp

from recserve.eval.metrics import (
    hit_rate_at_k,
    hits_matrix,
    metrics_from_hits,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

RANKED = [10, 20, 30, 40, 50]
RELEVANT = {20, 50, 99}


def test_recall_hand_computed():
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, RELEVANT, 100) == pytest.approx(2 / 3)  # k beyond list length


def test_hit_rate_and_mrr_hand_computed():
    assert hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0
    assert hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0
    assert reciprocal_rank_at_k(RANKED, RELEVANT, 5) == pytest.approx(1 / 2)
    assert reciprocal_rank_at_k(RANKED, RELEVANT, 1) == 0.0
    assert reciprocal_rank_at_k(RANKED, {10}, 5) == 1.0


def test_ndcg_hand_computed():
    # hits at ranks 2 and 5; |R| = 3 so the ideal list has 3 hits in the top 5
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(dcg / idcg)
    # ideal length is capped at k
    assert ndcg_at_k([20, 1, 2], RELEVANT, 1) == pytest.approx(1.0)


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    assert ndcg_at_k([4, 5, 6], {1, 2, 3}, 3) == 0.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 5)
    with pytest.raises(ValueError):
        ndcg_at_k(RANKED, RELEVANT, 0)
    with pytest.raises(ValueError):
        metrics_from_hits(np.zeros((1, 5), bool), np.array([0]), [5], 5)
    with pytest.raises(ValueError):
        metrics_from_hits(np.zeros((1, 5), bool), np.array([1]), [10], 5)


def test_hits_matrix():
    rel = sp.csr_matrix(np.array([[0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32))
    topk = np.array([[1, 2], [3, 0]])
    users = np.array([0, 1])
    np.testing.assert_array_equal(hits_matrix(topk, users, rel), [[True, False], [False, True]])
    # user order and selection must be respected
    np.testing.assert_array_equal(
        hits_matrix(np.array([[2, 3]]), np.array([2]), rel), [[True, False]]
    )


def test_vectorised_matches_reference():
    rng = np.random.default_rng(0)
    n_users, n_items, k_max = 50, 40, 20
    dense = rng.random((n_users, n_items)) < 0.1
    dense[np.arange(n_users), rng.integers(0, n_items, n_users)] = True  # >= 1 relevant each
    rel = sp.csr_matrix(dense.astype(np.float32))
    ranked = np.stack([rng.permutation(n_items)[:k_max] for _ in range(n_users)])
    users = np.arange(n_users)

    hits = hits_matrix(ranked, users, rel)
    got = metrics_from_hits(hits, dense.sum(axis=1), [5, 20], 10)

    rel_sets = [set(np.flatnonzero(row)) for row in dense]

    def mean_of(fn, k):
        return np.mean([fn(list(r), s, k) for r, s in zip(ranked, rel_sets, strict=True)])

    ref = {
        "recall@5": mean_of(recall_at_k, 5),
        "recall@20": mean_of(recall_at_k, 20),
        "ndcg@10": mean_of(ndcg_at_k, 10),
        "mrr@10": mean_of(reciprocal_rank_at_k, 10),
        "hitrate@10": mean_of(hit_rate_at_k, 10),
    }
    for name, value in ref.items():
        assert got[name] == pytest.approx(value), name
