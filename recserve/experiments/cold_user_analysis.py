"""M1: why popularity scores so well on cold users (MovieLens signup ratings).

New MovieLens users rate a batch of movies right after signing up, mostly
well-known titles. This script measures that burst so offline cold-start
numbers are read with the right caveat.

Usage::

    python -m recserve.experiments.cold_user_analysis --config configs/default.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from recserve.config import config_from_cli, run_tag
from recserve.data.prepare import load_split
from recserve.utils import log, setup_logging, write_result

FIRST_WINDOW_S = 3600
TOP_N = 100


def analyse(history: pd.DataFrame, period: pd.DataFrame) -> dict[str, Any]:
    warm_ids = set(history.loc[history["label"] == 1, "user_idx"])
    pos = period[period["label"] == 1]
    top = set(history[history["label"] == 1].groupby("item_idx").size().nlargest(TOP_N).index)
    first_ts = period.groupby("user_idx")["ts"].transform("min")
    in_first = (period["ts"] - first_ts) < FIRST_WINDOW_S

    out = {}
    for kind, is_warm in (("cold", False), ("warm", True)):
        users = set(pos["user_idx"]) & warm_ids if is_warm else set(pos["user_idx"]) - warm_ids
        rows = period["user_idx"].isin(users)
        upos = pos[pos["user_idx"].isin(users)]
        out[kind] = {
            "users": len(users),
            "median_ratings_in_period": float(period[rows].groupby("user_idx").size().median()),
            "share_ratings_in_first_hour": round(float(in_first[rows].mean()), 4),
            f"share_positives_in_history_top{TOP_N}": round(
                float(upos["item_idx"].isin(top).mean()), 4
            ),
        }
    return out


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    parts, _ = load_split(cfg["paths"]["split_dir"])
    result = {
        "val": analyse(parts["train"], parts["val"]),
        "test": analyse(pd.concat([parts["train"], parts["val"]]), parts["test"]),
    }
    log.info("%s", result)
    write_result(
        Path(cfg["paths"]["results_dir"]) / f"m1_cold_users_{run_tag(cfg)}.json",
        {"first_window_s": FIRST_WINDOW_S, "top_n": TOP_N, "results": result},
        cfg,
    )


if __name__ == "__main__":
    main()
