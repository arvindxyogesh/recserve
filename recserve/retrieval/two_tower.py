"""Two-tower retrieval model trained with in-batch sampled softmax.

User tower: masked mean of the ID embeddings of the user's last L positives,
plus their mean genre profile and log history length, through an MLP. There
is no user-ID embedding, so the tower is inductive: any user with history
gets an embedding, and it changes as soon as their history does.

Item tower: item-ID embedding, genre multi-hot, release-year bucket and the
genome tag vector (projected), through an MLP. Content features let items
with few interactions still land near similar items.

Both towers output L2-normalised vectors, so the score is cosine similarity
and the index can use inner product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from recserve.data.item_features import ItemFeatures


def mlp(n_in: int, hidden: int, n_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_out)
    )


class TwoTower(nn.Module):
    def __init__(
        self, feats: ItemFeatures, emb_dim: int = 64, hidden: int = 256, dropout: float = 0.1
    ) -> None:
        super().__init__()
        n_items = feats.n_items
        self.pad = n_items  # history padding index
        self.item_id = nn.Embedding(n_items + 1, emb_dim, padding_idx=self.pad)
        nn.init.normal_(self.item_id.weight, std=0.01)
        with torch.no_grad():
            self.item_id.weight[self.pad].zero_()

        pad_row = lambda a: np.concatenate([a, np.zeros((1, *a.shape[1:]), a.dtype)])  # noqa: E731
        self.register_buffer("genres", torch.from_numpy(pad_row(feats.genres)))
        self.register_buffer("year", torch.from_numpy(pad_row(feats.year_bucket)))
        self.register_buffer("has_genome", torch.from_numpy(pad_row(feats.has_genome))[:, None])
        n_genres = feats.genres.shape[1]
        self.year_emb = nn.Embedding(feats.n_year_buckets, 16)
        self.use_genome = feats.genome is not None
        genome_dim = 0
        if self.use_genome:
            self.register_buffer("genome", torch.from_numpy(pad_row(feats.genome)))
            self.genome_proj = nn.Linear(feats.genome.shape[1], emb_dim)
            genome_dim = emb_dim + 1

        self.item_mlp = mlp(emb_dim + n_genres + 16 + genome_dim, hidden, emb_dim, dropout)
        self.user_mlp = mlp(emb_dim + n_genres + 1, hidden, emb_dim, dropout)

    def item_embed(self, items: torch.Tensor) -> torch.Tensor:
        parts = [self.item_id(items), self.genres[items], self.year_emb(self.year[items])]
        if self.use_genome:
            parts += [self.genome_proj(self.genome[items]), self.has_genome[items]]
        return F.normalize(self.item_mlp(torch.cat(parts, dim=-1)), dim=-1)

    def user_embed(self, history: torch.Tensor) -> torch.Tensor:
        """history: (B, L) item indices, right-aligned, padded with self.pad."""
        mask = (history != self.pad).float().unsqueeze(-1)
        n = mask.sum(dim=1).clamp(min=1.0)
        pooled_id = (self.item_id(history) * mask).sum(dim=1) / n
        pooled_genre = (self.genres[history] * mask).sum(dim=1) / n
        x = torch.cat([pooled_id, pooled_genre, torch.log1p(n)], dim=-1)
        return F.normalize(self.user_mlp(x), dim=-1)


def in_batch_softmax_loss(
    user: torch.Tensor,
    item: torch.Tensor,
    item_ids: torch.Tensor,
    temperature: float,
    log_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sampled softmax where the other positives in the batch are the negatives.

    log_q (log sampling probability of each in-batch item) is subtracted from
    the logits: popular items appear as negatives more often, and without the
    correction the model learns to push them down (Yi et al., RecSys 2019).
    Duplicate items in the batch are masked so a user's positive is never
    also treated as their negative.
    """
    logits = user @ item.T / temperature
    if log_q is not None:
        logits = logits - log_q[None, :]
    same = item_ids[None, :] == item_ids[:, None]
    eye = torch.eye(len(item_ids), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    labels = torch.arange(len(item_ids), device=logits.device)
    return F.cross_entropy(logits, labels)


@dataclass
class Sequences:
    """All positives sorted by (user, ts); training targets index into ``items``."""

    items: np.ndarray  # (N,) item_idx, grouped by user, chronological within a user
    user_of: np.ndarray  # (N,) user_idx of each position
    start_of: np.ndarray  # (N,) first position of that user's block
    user_start: np.ndarray  # (n_users + 1,) block offsets per user (CSR style)

    @classmethod
    def from_positives(cls, pos: pd.DataFrame, n_users: int) -> Sequences:
        pos = pos.sort_values(["user_idx", "ts", "item_idx"], kind="stable")
        users = pos["user_idx"].to_numpy()
        counts = np.bincount(users, minlength=n_users)
        user_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return cls(
            items=pos["item_idx"].to_numpy().astype(np.int64),
            user_of=users.astype(np.int64),
            start_of=user_start[users],
            user_start=user_start,
        )

    def targets(self) -> np.ndarray:
        """Positions that have at least one earlier positive to use as history."""
        return np.flatnonzero(np.arange(len(self.items)) > self.start_of)


def history_matrix(
    items: torch.Tensor, start: torch.Tensor, end: torch.Tensor, length: int, pad: int
) -> torch.Tensor:
    """Right-aligned last ``length`` items of items[start:end] per row, padded with ``pad``."""
    idx = end[:, None] - length + torch.arange(length, device=items.device)[None, :]
    valid = idx >= start[:, None]
    return torch.where(valid, items[idx.clamp(min=0)], torch.full_like(idx, pad))
