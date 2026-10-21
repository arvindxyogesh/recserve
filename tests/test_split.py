import numpy as np
import pandas as pd
import pytest

from recserve.data.prepare import build_interactions, split_stats, temporal_split
from recserve.data.synthetic import make_synthetic


@pytest.fixture(scope="module")
def interactions():
    data = make_synthetic(n_users=200, n_items=120, seed=1)
    df, user_map, item_map = build_interactions(data["ratings"], data["movies"], 4.0)
    return data, df, user_map, item_map


def test_indices_are_contiguous_and_complete(interactions):
    data, df, user_map, item_map = interactions
    assert len(df) == len(data["ratings"])
    assert item_map["item_idx"].tolist() == list(range(len(data["movies"])))
    assert user_map["user_idx"].tolist() == list(range(data["ratings"]["userId"].nunique()))
    assert df["item_idx"].max() < len(item_map)


def test_label_threshold(interactions):
    _, df, _, _ = interactions
    np.testing.assert_array_equal(df["label"].to_numpy(), (df["rating"] >= 4.0).astype(int))


def test_temporal_split_is_disjoint_and_ordered(interactions):
    _, df, _, _ = interactions
    parts, cut = temporal_split(df, 0.8, 0.9)
    assert sum(len(p) for p in parts.values()) == len(df)
    assert parts["train"]["ts"].max() < cut["t_val"] <= parts["val"]["ts"].min()
    assert parts["val"]["ts"].max() < cut["t_test"] <= parts["test"]["ts"].min()
    # no (user, item) pair appears in more than one split
    keys = [set(zip(p["user_idx"], p["item_idx"], strict=True)) for p in parts.values()]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])


def test_split_stats_warm_cold(interactions):
    _, df, _, _ = interactions
    parts, cut = temporal_split(df, 0.8, 0.9)
    stats = split_stats(parts, cut)
    for name in ("val", "test"):
        s = stats[name]
        assert s["warm_eval_users"] + s["cold_eval_users"] == s["users_with_positive"]


def test_bad_quantiles(interactions):
    _, df, _, _ = interactions
    with pytest.raises(ValueError):
        temporal_split(df, 0.9, 0.8)


def test_user_sampling_is_seeded():
    data = make_synthetic(n_users=100, n_items=50, seed=2)
    a, _, _ = build_interactions(data["ratings"], data["movies"], 4.0, 30, np.random.default_rng(7))
    b, _, _ = build_interactions(data["ratings"], data["movies"], 4.0, 30, np.random.default_rng(7))
    assert a["user_idx"].nunique() == 30
    pd.testing.assert_frame_equal(a, b)
