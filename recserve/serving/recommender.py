"""Request-path logic: user tower -> HNSW retrieval -> LambdaRank, plus cold start.

Pure, synchronous compute with no web framework or Redis, so it can be unit
tested and benchmarked on its own. Large read-only tables are memory-mapped,
so several worker processes share one copy through the OS page cache.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd

from recserve.data.history import PAD
from recserve.ranking.features import ITEM_STATS, USER_STATS, PairFeaturizer, assemble
from recserve.serving.store import UserRecord
from recserve.serving.user_tower import NumpyUserTower

POP_90D = ITEM_STATS.index("i_pop_90d")


@dataclass
class Recommendation:
    items: np.ndarray  # item_idx, best first
    scores: np.ndarray
    retrieval_rank: np.ndarray | None  # 1-based rank in the candidate set
    source: str
    timings_s: dict[str, float] = field(default_factory=dict)


def right_align(recent: np.ndarray, length: int) -> np.ndarray:
    """(1, length) matrix of the last ``length`` items, padded on the left with PAD."""
    out = np.full((1, length), PAD, dtype=np.int64)
    tail = recent[-length:]
    if len(tail):
        out[0, -len(tail) :] = tail
    return out


class TwoStageRecommender:
    def __init__(self, art_dir: str | Path, threads: int = 1) -> None:
        art = Path(art_dir)
        meta = json.loads((art / "meta.json").read_text())
        scfg = meta["serving_config"]
        self.n_candidates = scfg["candidates"]
        self.ef_search = scfg["ef_search"]
        self.max_search = scfg["max_search"]
        self.tower_len = meta["two_tower_history_len"]
        self.rank_len = meta["ranker_history_len"]
        self.genre_names = meta["genre_names"]

        faiss.omp_set_num_threads(threads)
        self.index = faiss.read_index(str(art / "items.hnsw"))
        self.tower = NumpyUserTower(art / "user_tower.npz")
        self.booster = lgb.Booster(model_file=str(art / "ranker.txt"))
        self.item_stats = np.load(art / "item_stats.npy")
        self.pair = PairFeaturizer(
            genres=np.load(art / "item_genres.npy"),
            year=np.load(art / "item_year.npy"),
            has_genome=np.load(art / "item_has_genome.npy"),
            genome_unit=np.load(art / "genome_unit.npy", mmap_mode="r"),
        )
        items = pd.read_parquet(art / "items.parquet")
        self.movie_ids = items["movieId"].to_numpy()
        self.titles = items["title"].fillna("").to_numpy()
        cold = json.loads((art / "cold_start.json").read_text())
        self.cold_global = np.array(cold["global"], dtype=np.int64)
        self.cold_by_genre = {g: np.array(v, dtype=np.int64) for g, v in cold["by_genre"].items()}

    def warmup(self) -> None:
        """Pay one-off costs at startup instead of on the first requests.

        Reading the memory-mapped genome table once pulls it into the OS page
        cache (shared by all workers), so requests do not take page faults; a
        dummy request initialises LightGBM and FAISS search state.
        """
        genome = self.pair.genome_unit
        for start in range(0, len(genome), 4096):
            np.asarray(genome[start : start + 4096]).sum()
        dummy = UserRecord(
            recent=self.cold_global[:5],
            seen=self.cold_global[:5],
            stats=np.zeros(len(USER_STATS), np.float32),
        )
        self.recommend(dummy, 10)

    def retrieve(self, record: UserRecord) -> tuple[np.ndarray, np.ndarray]:
        """Top candidates by inner product, with already-rated items filtered out.

        Over-fetches by the number of seen items (capped) so filtering still
        leaves enough candidates; efSearch is passed per query, which is
        thread-safe, instead of being set on the shared index.
        """
        query = self.tower.embed(right_align(record.recent, self.tower_len))
        k_search = min(self.n_candidates + len(record.seen), self.max_search, self.index.ntotal)
        params = faiss.SearchParametersHNSW()
        params.efSearch = max(self.ef_search, k_search)
        scores, ids = self.index.search(query, k_search, params=params)
        keep = (ids[0] >= 0) & ~np.isin(ids[0], record.seen)
        return ids[0][keep][: self.n_candidates], scores[0][keep][: self.n_candidates]

    def rank(self, record: UserRecord, items: np.ndarray, scores: np.ndarray) -> np.ndarray:
        n = len(items)
        x = assemble(
            np.repeat(record.stats[None, :], n, axis=0),
            self.item_stats[items],
            self.pair(right_align(record.recent, self.rank_len), items[None, :], scores[None, :]),
        )
        return self.booster.predict(x, num_threads=1)

    def recommend(self, record: UserRecord, k: int) -> Recommendation:
        t0 = time.perf_counter()
        items, scores = self.retrieve(record)
        t1 = time.perf_counter()
        ranked = self.rank(record, items, scores) if len(items) else np.zeros(0)
        order = np.argsort(-ranked, kind="stable")[:k]
        t2 = time.perf_counter()
        return Recommendation(
            items=items[order],
            scores=ranked[order],
            retrieval_rank=order + 1,
            source="two_stage",
            timings_s={"retrieval": t1 - t0, "ranking": t2 - t1},
        )

    def cold_start(
        self, k: int, genres: list[str] | None = None, seen: np.ndarray | None = None
    ) -> Recommendation:
        """Recent popularity, interleaved across the requested genres if any."""
        t0 = time.perf_counter()
        if genres:
            lists = [self.cold_by_genre[g] for g in genres]
            depth = max(len(lst) for lst in lists)
            merged = [lst[i] for i in range(depth) for lst in lists if i < len(lst)]
            pool = np.array(list(dict.fromkeys(merged)), dtype=np.int64)
        else:
            pool = self.cold_global
        if seen is not None and len(seen):
            pool = pool[~np.isin(pool, seen)]
        items = pool[:k]
        return Recommendation(
            items=items,
            scores=self.item_stats[items, POP_90D].astype(np.float64),
            retrieval_rank=None,
            source="cold_start_genre" if genres else "cold_start_global",
            timings_s={"retrieval": time.perf_counter() - t0, "ranking": 0.0},
        )
