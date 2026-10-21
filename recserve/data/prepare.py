"""Index users and items, label positives and make the global temporal split.

Split protocol (see README, "Data and split"):
  * Two global timestamp cutoffs, t_val and t_test, are quantiles of the
    timestamps of all ratings. train = [start, t_val), val = [t_val, t_test),
    test = [t_test, end].
  * A single global cutoff (rather than a per-user last-N split) means no
    model or feature ever sees an interaction from after the period being
    evaluated, including other users' interactions. This rules out the
    "future popularity" leakage of per-user leave-last-out splits. Every
    user's val/test interactions are still later than their train ones.
  * Ratings >= positive_threshold are positives. All ratings are kept, because
    negatives are still useful for masking already-seen items and for ranking
    features.

Usage::

    python -m recserve.data.prepare --config configs/default.yaml
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recserve.config import config_from_cli, run_tag
from recserve.utils import log, set_seed, setup_logging, timed, write_result

SPLITS = ("train", "val", "test")


def build_interactions(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    positive_threshold: float,
    sample_users: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map raw ids to contiguous indices and add the binary label.

    Items are indexed over the whole catalogue (movies table), not just rated
    movies, so item indices are stable across samples and splits.
    """
    if sample_users is not None:
        rng = rng or np.random.default_rng(0)
        users = ratings["userId"].unique()
        keep = rng.choice(users, size=min(sample_users, len(users)), replace=False)
        ratings = ratings[ratings["userId"].isin(keep)]

    item_map = (
        movies[["movieId"]].drop_duplicates().sort_values("movieId").reset_index(drop=True)
    )
    item_map["item_idx"] = np.arange(len(item_map), dtype=np.int32)
    user_map = pd.DataFrame({"userId": np.sort(ratings["userId"].unique())})
    user_map["user_idx"] = np.arange(len(user_map), dtype=np.int32)

    df = ratings.merge(user_map, on="userId").merge(item_map, on="movieId")
    if len(df) != len(ratings):
        raise ValueError(f"{len(ratings) - len(df)} ratings reference unknown movies")
    out = pd.DataFrame({
        "user_idx": df["user_idx"].astype(np.int32),
        "item_idx": df["item_idx"].astype(np.int32),
        "rating": df["rating"].astype(np.float32),
        "ts": df["timestamp"].astype(np.int64),
        "label": (df["rating"] >= positive_threshold).astype(np.int8),
    })
    out = out.sort_values(["ts", "user_idx", "item_idx"], kind="stable").reset_index(drop=True)
    return out, user_map, item_map


def temporal_split(
    df: pd.DataFrame, val_quantile: float, test_quantile: float
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    if not 0 < val_quantile < test_quantile < 1:
        raise ValueError("need 0 < val_quantile < test_quantile < 1")
    t_val, t_test = (int(x) for x in np.quantile(df["ts"].to_numpy(), [val_quantile,
                                                                       test_quantile]))
    ts = df["ts"]
    parts = {
        "train": df[ts < t_val],
        "val": df[(ts >= t_val) & (ts < t_test)],
        "test": df[ts >= t_test],
    }
    return {k: v.reset_index(drop=True) for k, v in parts.items()}, {"t_val": t_val,
                                                                    "t_test": t_test}


def split_stats(parts: dict[str, pd.DataFrame], cutoffs: dict[str, int]) -> dict[str, Any]:
    """Row, user and item counts per split, plus warm/cold user counts.

    A user is *warm* for an evaluation period if they have at least one
    positive before it (train for val, train+val for test) and *evaluable* if
    they also have at least one positive inside it.
    """
    stats: dict[str, Any] = {
        "cutoffs": {k: {"unix": v, "utc": str(pd.Timestamp(v, unit="s"))}
                    for k, v in cutoffs.items()},
        "time_range_utc": {
            "start": str(pd.Timestamp(int(parts["train"]["ts"].min()), unit="s")),
            "end": str(pd.Timestamp(int(parts["test"]["ts"].max()), unit="s")),
        },
    }
    for name, part in parts.items():
        pos = part[part["label"] == 1]
        stats[name] = {
            "ratings": len(part),
            "positives": len(pos),
            "users": int(part["user_idx"].nunique()),
            "items": int(part["item_idx"].nunique()),
        }
    hist_pos: set[int] = set(parts["train"].loc[parts["train"]["label"] == 1, "user_idx"])
    for name in ("val", "test"):
        part = parts[name]
        eval_users = set(part.loc[part["label"] == 1, "user_idx"])
        stats[name]["users_with_positive"] = len(eval_users)
        stats[name]["warm_eval_users"] = len(eval_users & hist_pos)
        stats[name]["cold_eval_users"] = len(eval_users - hist_pos)
        hist_pos |= eval_users  # test history = train + val
    return stats


def load_split(split_dir: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    split_dir = Path(split_dir)
    parts = {s: pd.read_parquet(split_dir / f"{s}.parquet") for s in SPLITS}
    meta = json.loads((split_dir / "split_meta.json").read_text())
    return parts, meta


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    rng = set_seed(cfg["seed"])
    processed = Path(cfg["paths"]["processed_dir"])
    split_dir = Path(cfg["paths"]["split_dir"])
    split_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    with timed("load", timings):
        ratings = pd.read_parquet(processed / "ratings.parquet")
        movies = pd.read_parquet(processed / "movies.parquet")
    with timed("index", timings):
        df, user_map, item_map = build_interactions(
            ratings, movies, cfg["data"]["positive_threshold"], cfg["data"]["sample_users"], rng
        )
    with timed("split", timings):
        parts, cutoffs = temporal_split(
            df, cfg["split"]["val_quantile"], cfg["split"]["test_quantile"]
        )
        stats = split_stats(parts, cutoffs)

    for name, part in parts.items():
        part.to_parquet(split_dir / f"{name}.parquet", index=False)
    user_map.to_parquet(split_dir / "user_map.parquet", index=False)
    item_map.to_parquet(split_dir / "item_map.parquet", index=False)
    meta = {
        "n_users": len(user_map),
        "n_items": len(item_map),
        "positive_threshold": cfg["data"]["positive_threshold"],
        "sample_users": cfg["data"]["sample_users"],
        **stats,
    }
    (split_dir / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_result(Path(cfg["paths"]["results_dir"]) / f"data_split_{run_tag(cfg)}.json",
                 {"split": meta, "timings_s": timings}, cfg)
    log.info("split summary:\n%s", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
