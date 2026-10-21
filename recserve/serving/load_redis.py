"""Load per-user serving state (as of the start of the test period) into Redis.

Usage::

    python -m recserve.serving.load_redis --config configs/default.yaml \
        --set serving.redis_url=redis://localhost:6379/0
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import redis

from recserve.config import config_from_cli
from recserve.data.history import RecentHistory
from recserve.data.prepare import load_split
from recserve.ranking.features import user_stats
from recserve.serving.store import KEY, UserRecord, encode
from recserve.utils import log, setup_logging, timed

BATCH = 2000


@dataclass
class UserState:
    """Everything the API knows about users at serving time ("now" = t_test)."""

    users: np.ndarray  # internal user_idx with at least one rating before t_test
    external: np.ndarray  # userId by user_idx
    recent: RecentHistory
    hist_len: int
    seen_items: np.ndarray
    seen_start: np.ndarray
    stats: np.ndarray  # (n_users, len(USER_STATS))
    parts: dict[str, pd.DataFrame]
    meta: dict[str, Any]

    def record(self, u: int) -> UserRecord:
        rec = self.recent.matrix(np.array([u]), self.hist_len)[0]
        return UserRecord(
            recent=rec[rec >= 0],
            seen=self.seen_items[self.seen_start[u] : self.seen_start[u + 1]],
            stats=self.stats[u],
        )


def build_user_state(cfg: dict[str, Any]) -> UserState:
    split_dir = Path(cfg["paths"]["split_dir"])
    parts, meta = load_split(split_dir)
    history = pd.concat([parts["train"], parts["val"]], ignore_index=True)
    n_users = meta["n_users"]
    seen = history.sort_values("user_idx", kind="stable")
    counts = np.bincount(seen["user_idx"], minlength=n_users)
    seen_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    user_map = pd.read_parquet(split_dir / "user_map.parquet").sort_values("user_idx")
    return UserState(
        users=np.flatnonzero(counts > 0),
        external=user_map["userId"].to_numpy(),
        recent=RecentHistory.from_df(history, n_users),
        hist_len=max(cfg["ranker"]["history_len"], cfg["two_tower"]["history_len"]),
        seen_items=seen["item_idx"].to_numpy().astype(np.int64),
        seen_start=seen_start,
        stats=user_stats(history, n_users, meta["cutoffs"]["t_test"]["unix"]),
        parts=parts,
        meta=meta,
    )


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    url = os.environ.get("REDIS_URL", cfg["serving"]["redis_url"])
    state = build_user_state(cfg)
    users = state.users
    if cfg["serving"].get("max_users"):
        users = users[: cfg["serving"]["max_users"]]
    client = redis.Redis.from_url(url)
    client.ping()
    with timed(f"load {len(users)} users into {url}"):
        for b in range(0, len(users), BATCH):
            pipe = client.pipeline(transaction=False)
            for u in users[b : b + BATCH]:
                r = state.record(u)
                pipe.hset(
                    KEY.format(int(state.external[u])), mapping=encode(r.recent, r.seen, r.stats)
                )
            pipe.execute()
    client.set("meta:as_of_ts", state.meta["cutoffs"]["t_test"]["unix"])
    info = client.info("memory")
    log.info("redis used_memory_human=%s keys=%d", info["used_memory_human"], client.dbsize())


if __name__ == "__main__":
    main()
