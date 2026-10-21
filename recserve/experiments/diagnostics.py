"""Small diagnostics behind claims in the README.

  parity  numpy user tower vs the torch embeddings exported at training time
          (host; needs the train extra)
  redis   async Redis client HGETALL latency at several concurrency levels
          (run in the serving container next to Redis, see `make diag-redis`)

Usage::

    python -m recserve.experiments.diagnostics parity
    python -m recserve.experiments.diagnostics redis --set serving.redis_url=redis://redis:6379/0
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recserve.config import add_config_args, load_config, run_tag
from recserve.utils import log, setup_logging, write_result


def parity(cfg: dict) -> dict:
    from recserve.data.history import RecentHistory
    from recserve.data.prepare import load_split
    from recserve.serving.user_tower import NumpyUserTower

    tag = run_tag(cfg)
    art = Path(cfg["paths"]["artifacts_dir"]) / "two_tower" / tag / "test"
    parts, meta = load_split(cfg["paths"]["split_dir"])
    history = pd.concat([parts["train"], parts["val"]])
    recent = RecentHistory.from_df(history, meta["n_users"])
    torch_emb = np.load(art / "user_emb.npy")
    has = np.load(art / "user_has_history.npy")
    users = np.random.default_rng(cfg["seed"]).choice(np.flatnonzero(has), 5000, replace=False)
    got = NumpyUserTower(art / "user_tower.npz").embed(
        recent.matrix(users, cfg["two_tower"]["history_len"])
    )
    return {
        "users": len(users),
        "max_abs_diff": float(np.abs(got - torch_emb[users]).max()),
        "min_cosine": float((got * torch_emb[users]).sum(axis=1).min()),
        "torch_device_at_export": "see artifacts/two_tower/<tag>/test/meta.json",
    }


async def _redis_bench(url: str, n: int, levels: list[int]) -> dict:
    import redis.asyncio as aioredis

    client = aioredis.from_url(url)
    keys = [k async for k in client.scan_iter("user:*", count=1000)][:n]
    out = {}
    for conc in levels:
        lat: list[float] = []

        async def one(key: bytes, lat: list[float] = lat) -> None:
            t0 = time.perf_counter()
            await client.hgetall(key)
            lat.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for i in range(0, len(keys), conc):
            await asyncio.gather(*(one(k) for k in keys[i : i + conc]))
        elapsed = time.perf_counter() - t0
        a = np.array(lat) * 1000
        out[f"concurrency_{conc}"] = {
            "ops_per_s": round(len(lat) / elapsed),
            "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
        }
    await client.aclose()
    return {"keys": len(keys), "client": "redis-py asyncio, no hiredis", "results": out}


def main() -> None:
    setup_logging()
    parser = add_config_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("check", choices=["parity", "redis"])
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.check == "parity":
        payload = parity(cfg)
    else:
        url = os.environ.get("REDIS_URL", cfg["serving"]["redis_url"])
        payload = asyncio.run(_redis_bench(url, 2000, [1, 8, 32]))
    log.info("%s", payload)
    write_result(
        Path(cfg["paths"]["results_dir"]) / f"diag_{args.check}_{run_tag(cfg)}.json",
        payload,
        cfg,
    )


if __name__ == "__main__":
    main()
