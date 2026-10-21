import numpy as np
import pandas as pd
import pytest

from recserve.eval.harness import EvalContext, evaluate, top_k
from recserve.retrieval.baselines import CooccurrenceRecommender, PopularityRecommender


def frame(rows):
    df = pd.DataFrame(rows, columns=["user_idx", "item_idx", "ts", "label"])
    df["rating"] = np.where(df["label"] == 1, 5.0, 2.0).astype(np.float32)
    return df


@pytest.fixture
def ctx():
    # users 0-2 like items {0, 1}; user 3 likes {2, 3}; item 4 only liked recently
    history = frame(
        [
            (0, 0, 1, 1),
            (0, 1, 2, 1),
            (1, 0, 3, 1),
            (1, 1, 4, 1),
            (2, 0, 5, 1),
            (3, 2, 6, 1),
            (3, 3, 7, 1),
            (2, 4, 95, 1),
            (3, 4, 96, 1),
            (1, 3, 8, 0),
        ]
    )
    target = frame([(2, 1, 101, 1), (3, 0, 102, 1), (0, 3, 103, 0)])
    return EvalContext("val", history, target, n_users=4, n_items=6, start_ts=100)


def test_popularity_all_time_and_recent(ctx):
    pop = PopularityRecommender()
    pop.fit(ctx)
    np.testing.assert_array_equal(pop.item_scores, [3, 2, 1, 1, 2, 0])
    recent = PopularityRecommender(window_days=1)
    recent.fit(ctx)
    np.testing.assert_array_equal(recent.item_scores, [3, 2, 1, 1, 2, 0])  # all within 1 day
    ctx.start_ts = 10 * 86_400
    recent.fit(ctx)
    assert recent.item_scores.sum() == 0


def test_cooccurrence_recommends_co_liked_item(ctx):
    model = CooccurrenceRecommender(top_neighbors=3, shrinkage=0.0, min_item_positives=1)
    model.fit(ctx)
    # items 0 and 1 are co-liked by two users, so they must be each other's neighbours
    assert model.sim[0, 1] > 0 and model.sim[1, 0] > 0
    assert model.sim[0, 0] == 0  # no self-similarity
    scores = model.score(np.array([2]), ctx)
    assert scores[0, 1] > scores[0, 2]  # user 2 liked item 0, so item 1 beats item 2
    assert (scores[0][scores[0] <= 0] >= -1).all()  # backfill stays in [-1, 0]


def test_top_k_sorted():
    scores = np.array([[0.1, 0.9, 0.5, 0.7]], dtype=np.float32)
    np.testing.assert_array_equal(top_k(scores, 3), [[1, 3, 2]])


def test_evaluate_masks_history_and_counts_users(ctx):
    res = evaluate(PopularityRecommender(), ctx, retrieval_ks=[2, 3], ranking_k=2)
    # users 2 and 3 have a target positive and a history positive; user 0's target is negative
    assert res["n_users"] == 2
    # user 2 already rated items 0 and 4, so item 1 (next most popular) must be ranked first
    # and counts as a hit; user 3's target item 0 is the most popular unseen item
    assert res["metrics"]["hitrate@2"] == 1.0
    assert res["metrics"]["mrr@2"] == 1.0
