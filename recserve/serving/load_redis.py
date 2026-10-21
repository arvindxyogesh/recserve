"""Load per-user serving state (as of the start of the test period) into Redis.

Usage::

    python -m recserve.serving.load_redis --config configs/default.yaml \
        --set serving.redis_url=redis://localhost:6379/0
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import redis

from recserve.config import config_from_cli
from recserve.data.history import RecentHistory
from recserve.data.prepare import load_split
from recserve.ranking.features import user_stats
from recserve.serving.store import KEY, encode
from recserve.utils import log, setup_logging, timed

BATCH = 2000


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    scfg = cfg["serving"]
    url = os.environ.get("REDIS_URL", scfg["redis_url"])
    split_dir = Path(cfg["paths"]["split_dir"])
    parts, meta = load_split(split_dir)
    history = pd.concat([parts["train"], parts["val"]], ignore_index=True)
    as_of = meta["cutoffs"]["t_test"]["unix"]
    user_map = pd.read_parquet(split_dir / "user_map.parquet")
    n_users = meta["n_users"]

    stats = user_stats(history, n_users, as_of)
    hist_len = max(cfg["ranker"]["history_len"], cfg["two_tower"]["history_len"])
    recent = RecentHistory.from_df(history, n_users)
    seen = history.sort_values("user_idx", kind="stable")
    seen_items = seen["item_idx"].to_numpy()
    seen_start = np.concatenate([[0], np.cumsum(np.bincount(seen["user_idx"], minlength=n_users))])

    users = np.flatnonzero(np.diff(seen_start) > 0)  # anyone with a rating before t_test
    if scfg.get("max_users"):
        users = users[: scfg["max_users"]]
    external = user_map.set_index("user_idx")["userId"]
    client = redis.Redis.from_url(url)
    client.ping()
    with timed(f"load {len(users)} users into {url}"):
        for b in range(0, len(users), BATCH):
            pipe = client.pipeline(transaction=False)
            batch = users[b : b + BATCH]
            rec = recent.matrix(batch, hist_len)
            for row, u in zip(rec, batch, strict=True):
                pipe.hset(
                    KEY.format(int(external[u])),
                    mapping=encode(
                        row[row >= 0], seen_items[seen_start[u] : seen_start[u + 1]], stats[u]
                    ),
                )
            pipe.execute()
    client.set("meta:as_of_ts", as_of)
    info = client.info("memory")
    log.info("redis used_memory_human=%s keys=%d", info["used_memory_human"], client.dbsize())


if __name__ == "__main__":
    main()
