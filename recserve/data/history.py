"""Per-user recent positive history in numpy (torch-free, so serving can use it)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PAD = -1


@dataclass
class RecentHistory:
    items: np.ndarray  # positives grouped by user, chronological within a user
    user_start: np.ndarray  # (n_users + 1,) CSR-style offsets

    @classmethod
    def from_df(cls, history: pd.DataFrame, n_users: int) -> RecentHistory:
        pos = history[history["label"] == 1].sort_values(
            ["user_idx", "ts", "item_idx"], kind="stable"
        )
        counts = np.bincount(pos["user_idx"].to_numpy(), minlength=n_users)
        start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return cls(pos["item_idx"].to_numpy().astype(np.int64), start)

    def matrix(self, users: np.ndarray, length: int) -> np.ndarray:
        """(len(users), length) last positives per user, right-aligned, padded with PAD."""
        start, end = self.user_start[users], self.user_start[users + 1]
        idx = end[:, None] - length + np.arange(length)[None, :]
        valid = idx >= start[:, None]
        return np.where(valid, self.items[np.clip(idx, 0, None)], PAD)
