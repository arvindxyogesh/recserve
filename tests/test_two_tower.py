import numpy as np
import pandas as pd
import pytest
import torch

from recserve.data.item_features import ItemFeatures, year_bucket
from recserve.retrieval.two_tower import (
    Sequences,
    TwoTower,
    history_matrix,
    in_batch_softmax_loss,
)


def tiny_features(n_items=10, genome=True):
    rng = np.random.default_rng(0)
    return ItemFeatures(
        genres=(rng.random((n_items, 3)) < 0.5).astype(np.float32),
        genre_names=["a", "b", "c"],
        year_bucket=rng.integers(0, 5, n_items),
        n_year_buckets=5,
        genome=rng.random((n_items, 4)).astype(np.float32) if genome else None,
        has_genome=np.ones(n_items, dtype=np.float32),
    )


def test_year_bucket():
    assert year_bucket("Toy Story (1995)") == 1 + (1995 - 1900) // 5
    assert year_bucket("No year here") == 0
    assert year_bucket("Old (1850)") == 1  # clipped to the first bucket


def test_history_matrix_right_aligned_and_padded():
    items = torch.tensor([5, 6, 7, 8, 9])
    start = torch.tensor([0, 0, 3])
    end = torch.tensor([1, 3, 5])
    out = history_matrix(items, start, end, length=3, pad=-1)
    assert out.tolist() == [[-1, -1, 5], [5, 6, 7], [-1, 8, 9]]


def test_sequences_targets_skip_first_positive():
    pos = pd.DataFrame(
        {"user_idx": [1, 0, 0, 1, 0], "item_idx": [3, 1, 2, 4, 0], "ts": [10, 3, 1, 20, 2]}
    )
    seq = Sequences.from_positives(pos, n_users=3)
    assert seq.items.tolist() == [2, 0, 1, 3, 4]  # user 0 chronological, then user 1
    assert seq.user_start.tolist() == [0, 3, 5, 5]
    assert seq.targets().tolist() == [1, 2, 4]  # first position of each user has no history


@pytest.mark.parametrize("genome", [True, False])
def test_towers_output_unit_vectors(genome):
    model = TwoTower(tiny_features(genome=genome), emb_dim=8, hidden=16, dropout=0.0)
    items = model.item_embed(torch.arange(10))
    hist = torch.tensor([[model.pad, 1, 2], [3, 4, 5]])
    users = model.user_embed(hist)
    assert items.shape == (10, 8) and users.shape == (2, 8)
    torch.testing.assert_close(items.norm(dim=1), torch.ones(10))
    torch.testing.assert_close(users.norm(dim=1), torch.ones(2))


def test_padding_does_not_change_user_embedding():
    model = TwoTower(tiny_features(), emb_dim=8, hidden=16, dropout=0.0).eval()
    a = model.user_embed(torch.tensor([[1, 2]]))
    b = model.user_embed(torch.tensor([[model.pad, model.pad, 1, 2]]))
    torch.testing.assert_close(a, b)


def test_in_batch_loss_masks_duplicates_and_applies_logq():
    torch.manual_seed(0)
    u = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    v = u.clone()  # each user's positive is perfectly aligned
    ids = torch.tensor([0, 1, 2, 3])
    perfect = in_batch_softmax_loss(u, v, ids, temperature=0.05)
    assert perfect < 0.1

    # two copies of the same item: masking the duplicate must lower the loss
    dup_ids = torch.tensor([0, 0, 2, 3])
    v_dup = v.clone()
    v_dup[1] = v_dup[0]
    masked = in_batch_softmax_loss(u, v_dup, dup_ids, temperature=0.05)
    unmasked = in_batch_softmax_loss(u, v_dup, torch.arange(4), temperature=0.05)
    assert masked < unmasked

    # log_q is subtracted per column, before the softmax
    log_q = torch.tensor([0.0, 1.0, 2.0, 5.0])
    rand_v = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    expected = torch.nn.functional.cross_entropy(u @ rand_v.T / 0.1 - log_q, torch.arange(4))
    torch.testing.assert_close(in_batch_softmax_loss(u, rand_v, ids, 0.1, log_q), expected)


def test_training_reduces_loss():
    torch.manual_seed(0)
    feats = tiny_features(n_items=20)
    model = TwoTower(feats, emb_dim=8, hidden=16, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    hist = torch.randint(0, 20, (32, 5))
    tgt = hist[:, -1]  # learnable: target is the last history item
    losses = []
    for _ in range(100):
        loss = in_batch_softmax_loss(
            model.user_embed(hist[:, :-1]), model.item_embed(tgt), tgt, 0.1
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
