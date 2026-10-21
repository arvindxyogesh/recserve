"""Package everything the API loads into artifacts/serving/<tag>/.

"Now" for serving is the start of the test period: the test-period two-tower
model (trained on train + val), item statistics as of t_test, and the
ranker. User state is loaded into Redis separately (load_redis.py).

Usage::

    python -m recserve.serving.build_artifacts --config configs/default.yaml
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from recserve.config import config_from_cli, run_tag
from recserve.data.item_features import build_item_features
from recserve.data.prepare import load_split
from recserve.ranking.features import ITEM_STATS, PairFeaturizer, item_stats
from recserve.retrieval.ann import build_index
from recserve.utils import log, setup_logging, timed


def cold_start_lists(
    pop: np.ndarray, genres: np.ndarray, genre_names: list[str], size: int
) -> dict:
    """Most popular items overall and per genre (items with zero recent positives excluded)."""
    order = np.argsort(-pop, kind="stable")
    order = order[pop[order] > 0]
    by_genre = {
        name: [int(i) for i in order[genres[order, j] > 0][:size]]
        for j, name in enumerate(genre_names)
    }
    return {"global": [int(i) for i in order[:size]], "by_genre": by_genre}


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    scfg = cfg["serving"]
    tag = run_tag(cfg)
    art_root = Path(cfg["paths"]["artifacts_dir"])
    tt = art_root / "two_tower" / tag / "test"
    out = art_root / "serving" / tag
    out.mkdir(parents=True, exist_ok=True)
    split_dir = Path(cfg["paths"]["split_dir"])

    parts, meta = load_split(split_dir)
    history = pd.concat([parts["train"], parts["val"]], ignore_index=True)
    as_of = meta["cutoffs"]["t_test"]["unix"]
    item_map = pd.read_parquet(split_dir / "item_map.parquet")
    feats = build_item_features(cfg["paths"]["processed_dir"], item_map, use_genome=True)

    timings: dict[str, float] = {}
    with timed("hnsw index", timings):
        index, _ = build_index(
            np.load(tt / "item_emb.npy"), "hnsw", scfg["ann_m"], scfg["ef_construction"]
        )
        faiss.write_index(index, str(out / "items.hnsw"))
    shutil.copy(tt / "user_tower.npz", out / "user_tower.npz")
    shutil.copy(art_root / "ranker" / tag / "model.txt", out / "ranker.txt")

    istats = item_stats(history, meta["n_items"], as_of, feats)
    np.save(out / "item_stats.npy", istats)
    pair = PairFeaturizer.from_item_features(feats)
    np.save(out / "item_genres.npy", feats.genres)
    np.save(out / "item_year.npy", pair.year.astype(np.float32))
    np.save(out / "item_has_genome.npy", feats.has_genome)
    np.save(out / "genome_unit.npy", pair.genome_unit.astype(np.float32))

    pop = istats[:, ITEM_STATS.index("i_pop_90d")]
    lists = cold_start_lists(pop, feats.genres, feats.genre_names, scfg["cold_start_pool"])
    (out / "cold_start.json").write_text(json.dumps(lists) + "\n")

    movies = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "movies.parquet")
    items = item_map.merge(movies, on="movieId", how="left").sort_values("item_idx")
    items[["item_idx", "movieId", "title", "genres"]].to_parquet(out / "items.parquet", index=False)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "as_of_ts": as_of,
                "n_items": meta["n_items"],
                "genre_names": feats.genre_names,
                "serving_config": scfg,
                "ranker_history_len": cfg["ranker"]["history_len"],
                "two_tower_history_len": cfg["two_tower"]["history_len"],
                "build_timings_s": timings,
            },
            indent=2,
        )
        + "\n"
    )
    log.info("serving artifacts in %s", out)


if __name__ == "__main__":
    main()
