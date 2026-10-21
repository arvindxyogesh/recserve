import numpy as np
import pandas as pd
import pytest

from recserve.data.history import PAD, RecentHistory
from recserve.data.item_features import ItemFeatures
from recserve.eval.metrics import paired_bootstrap_ci, per_user_ranking
from recserve.ranking.features import (
    FEATURES,
    ITEM_STATS,
    PAIR,
    USER_STATS,
    PairFeaturizer,
    assemble,
    item_stats,
    user_stats,
)

DAY = 86_400


def history():
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1],
            "item_idx": [0, 1, 2, 1],
            "rating": [5.0, 2.0, 4.0, 4.5],
            "ts": [0, DAY, 2 * DAY, 9 * DAY],
            "label": [1, 0, 1, 1],
        }
    )


def feats():
    genome = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
    return ItemFeatures(
        genres=np.array([[1, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32),
        genre_names=["a", "b"],
        year_bucket=np.zeros(4, dtype=np.int64),
        n_year_buckets=1,
        genome=genome,
        has_genome=np.array([1, 1, 1, 0], dtype=np.float32),
        year=np.array([1990, 2000, np.nan, 2010], dtype=np.float32),
    )


def test_recent_history_matrix():
    rh = RecentHistory.from_df(history(), n_users=3)
    np.testing.assert_array_equal(
        rh.matrix(np.array([0, 1, 2]), 3), [[PAD, 0, 2], [PAD, PAD, 1], [PAD, PAD, PAD]]
    )


def test_user_stats():
    us = user_stats(history(), n_users=3, as_of_ts=10 * DAY)
    col = {c: i for i, c in enumerate(USER_STATS)}
    assert us[0, col["u_n_ratings"]] == 3 and us[0, col["u_n_pos"]] == 2
    assert us[0, col["u_mean_rating"]] == pytest.approx(11 / 3)
    assert us[0, col["u_days_since_last"]] == pytest.approx(8)
    assert us[0, col["u_days_active"]] == pytest.approx(2)
    assert np.isnan(us[2]).all()  # no history


def test_item_stats_windows():
    st = item_stats(history(), n_items=4, as_of_ts=10 * DAY, feats=feats())
    col = {c: i for i, c in enumerate(ITEM_STATS)}
    assert st[1, col["i_n_ratings"]] == 2 and st[1, col["i_n_pos"]] == 1
    assert st[1, col["i_pop_7d"]] == 1  # the positive on day 9 only
    assert st[0, col["i_pop_7d"]] == 0 and st[0, col["i_pop_90d"]] == 1
    assert st[3, col["i_n_ratings"]] == 0 and np.isnan(st[3, col["i_mean_rating"]])
    assert st[0, col["i_year"]] == 1990


def test_pair_features():
    pf = PairFeaturizer.from_item_features(feats())
    recent = np.array([[PAD, 0, 2]])  # items 0 (genre a, genome [1,0]) and 2 (a+b, [1,1])
    cand = np.array([[1, 3]])
    out = pf(recent, cand, np.array([[0.9, 0.5]]))
    col = {c: i for i, c in enumerate(PAIR)}
    assert out.shape == (2, len(PAIR))
    np.testing.assert_allclose(out[:, col["retrieval_rank"]], [1, 2])
    # genre profile = mean of [1,0] and [1,1] = [1, .5]; item 1 is genre b only
    assert out[0, col["genre_affinity"]] == pytest.approx(0.5)
    # genome profile ~ unit([1,0] + unit([1,1])); item 1 = [0,1]
    prof = np.array([1, 0]) + np.array([1, 1]) / np.sqrt(2)
    assert out[0, col["genome_sim"]] == pytest.approx(prof[1] / np.linalg.norm(prof), rel=1e-5)
    assert np.isnan(out[1, col["genome_sim"]])  # item 3 has no genome
    assert out[0, col["year_gap"]] == pytest.approx(10)  # 2000 vs mean(1990), item 2 year NaN


def test_assemble_order():
    x = assemble(
        np.zeros((2, len(USER_STATS))), np.ones((2, len(ITEM_STATS))), np.full((2, len(PAIR)), 2.0)
    )
    assert x.shape == (2, len(FEATURES))
    assert x[0, len(USER_STATS)] == 1 and x[0, -1] == 2


def test_per_user_and_bootstrap():
    hits = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=bool)
    pu = per_user_ranking(hits, np.array([1, 2, 1]), 3)
    np.testing.assert_allclose(pu["mrr"], [1, 0, 0.5])
    a, b = np.array([1.0, 1.0, 1.0, 1.0]), np.array([0.5, 0.5, 0.5, 0.5])
    ci = paired_bootstrap_ci(a, b, n_boot=200)
    assert ci["lift"] == pytest.approx(1.0) and ci["lift_ci"] == pytest.approx([1.0, 1.0])
