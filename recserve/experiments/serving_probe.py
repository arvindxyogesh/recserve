"""M4: sequential (single client) latency probe against a running API.

Not a load test (that is M5): one request at a time, random users who had a
rating before the test period, reporting the per-stage server-side timings
the API returns plus client-side round-trip time.

Usage::

    python -m recserve.experiments.serving_probe --set probe.url=http://localhost:8000
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from recserve.config import config_from_cli, run_tag
from recserve.utils import log, set_seed, setup_logging, write_result


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.load(resp)


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    rng = set_seed(cfg["seed"])
    pcfg = cfg["probe"]
    users = pd.read_parquet(Path(cfg["paths"]["split_dir"]) / "user_map.parquet")["userId"]
    ids = rng.choice(users.to_numpy(), size=pcfg["requests"], replace=False)

    rows = []
    for u in ids:
        t0 = time.perf_counter()
        r = get(f"{pcfg['url']}/recommend?user_id={u}&k={pcfg['k']}")
        rows.append(
            {
                "source": r["source"],
                "client_rtt": (time.perf_counter() - t0) * 1000,
                **r["latency_ms"],
            }
        )
    df = pd.DataFrame(rows)
    two = df[df["source"] == "two_stage"].drop(columns="source")
    pct = {
        col: {f"p{q}": round(float(np.percentile(two[col], q)), 3) for q in (50, 95, 99)}
        for col in two.columns
    }
    log.info("%s", json.dumps(pct, indent=1))
    write_result(
        Path(cfg["paths"]["results_dir"]) / f"m4_serving_probe_{run_tag(cfg)}.json",
        {
            "note": "sequential single-client requests; server-side stage timings in ms",
            "url": pcfg["url"],
            "requests": len(df),
            "sources": df["source"].value_counts().to_dict(),
            "two_stage_latency_ms": pct,
            "server": get(f"{pcfg['url']}/readyz"),
        },
        cfg,
    )


if __name__ == "__main__":
    main()
