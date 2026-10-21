"""User tower inference in numpy, so the serving image needs no torch.

The weights are exported from the trained TwoTower (see
recserve.retrieval.export_user_tower) and the forward pass below mirrors
TwoTower.user_embed exactly; a unit test checks the two agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from recserve.data.history import PAD


class NumpyUserTower:
    def __init__(self, path: str | Path) -> None:
        w = np.load(path)
        self.item_id = w["item_id"]  # (n_items + 1, d), last row = padding (zeros)
        self.genres = w["genres"]  # (n_items + 1, n_genres), last row zeros
        self.w0, self.b0 = w["w0"], w["b0"]
        self.w1, self.b1 = w["w1"], w["b1"]
        self.pad = self.item_id.shape[0] - 1

    def embed(self, recent: np.ndarray) -> np.ndarray:
        """recent: (B, L) item indices padded with PAD (-1). Returns (B, d) unit vectors."""
        idx = np.where(recent == PAD, self.pad, recent)
        mask = (idx != self.pad).astype(np.float32)[..., None]
        n = np.maximum(mask.sum(axis=1), 1.0)
        pooled_id = (self.item_id[idx] * mask).sum(axis=1) / n
        pooled_genre = (self.genres[idx] * mask).sum(axis=1) / n
        x = np.concatenate([pooled_id, pooled_genre, np.log1p(n)], axis=1)
        h = np.maximum(x @ self.w0.T + self.b0, 0.0)
        out = h @ self.w1.T + self.b1
        return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)
