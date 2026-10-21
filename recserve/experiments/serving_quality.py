"""M5: offline quality and in-process cost of the *serving* code path.

Runs TwoStageRecommender (HNSW retrieval + LambdaRank, exactly what the API
runs) for every warm test user, with user records built by the same code that
fills Redis, and scores the result against test positives. This checks that
serving reproduces the offline M3 numbers and measures what a serving
optimisation costs in quality. Must run where faiss and lightgbm can share a
process (Linux / the serving container).

Usage (inside the container, see `make serving-quality`)::

    python -m recserve.experiments.serving_quality --set serving.filter_mode=selector
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from recserve.config import config_from_cli, run_tag
from recserve.eval.harness import to_csr
from recserve.eval.metrics import hits_matrix, metrics_from_hits
from recserve.serving.load_redis import build_user_state
from recserve.serving.recommender import TwoStageRecommender
from recserve.utils import log, setup_logging, write_result


def pct(x: list[float]) -> dict[str, float]:
    a = np.array(x) * 1000
    return {f"p{q}": round(float(np.percentile(a, q)), 4) for q in (50, 95, 99)} | {
        "mean": round(float(a.mean()), 4)
    }


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    scfg = cfg["serving"]
    tag = run_tag(cfg)
    state = build_user_state(cfg)
    n_users, n_items = state.meta["n_users"], state.meta["n_items"]
    test = state.parts["test"]
    target = to_csr(test[test["label"] == 1], n_users, n_items)
    has_target = np.diff(target.indptr) > 0
    has_hist_pos = np.diff(state.recent.user_start) > 0
    users = np.flatnonzero(has_target & has_hist_pos)

    rec = TwoStageRecommender(
        Path(cfg["paths"]["artifacts_dir"]) / "serving" / tag,
        candidates=scfg["candidates"],
        filter_mode=scfg["filter_mode"],
    )
    rec.warmup()
    k, n = cfg["eval"]["ranking_k"], rec.n_candidates
    top = np.full((len(users), k), -1, dtype=np.int64)
    cands = np.full((len(users), n), -1, dtype=np.int64)
    t_ret, t_rank = [], []
    for row, u in enumerate(users):
        r = state.record(u)
        t0 = time.perf_counter()
        items, scores = rec.retrieve(r)
        t1 = time.perf_counter()
        order = np.argsort(-rec.rank(r, items, scores), kind="stable")[:k]
        t2 = time.perf_counter()
        t_ret.append(t1 - t0)
        t_rank.append(t2 - t1)
        cands[row, : len(items)] = items
        top[row, : len(order)] = items[order]

    n_rel = np.diff(target.indptr)[users]

    def hits(ids: np.ndarray) -> np.ndarray:
        return hits_matrix(np.where(ids < 0, 0, ids), users, target) & (ids >= 0)

    ranked = metrics_from_hits(hits(top), n_rel, [k], k)
    retrieval = {f"recall@{n}": float(np.mean(hits(cands).sum(axis=1) / n_rel))}
    variant = f"{scfg['filter_mode']}_c{n}"
    log.info("%s ranked %s retrieval %s", variant, ranked, retrieval)
    write_result(
        Path(cfg["paths"]["results_dir"]) / f"m5_quality_{tag}_{variant}.json",
        {
            "variant": {
                "filter_mode": rec.filter_mode,
                "candidates": n,
                "ef_search": rec.ef_search,
            },
            "n_users": int(len(users)),
            "ranked": ranked,
            "retrieval": retrieval,
            "in_process_ms": {
                "retrieval": pct(t_ret),
                "ranking": pct(t_rank),
                "total": pct([a + b for a, b in zip(t_ret, t_rank, strict=True)]),
            },
            "short_candidate_lists": int((cands < 0).any(axis=1).sum()),
        },
        cfg,
    )


if __name__ == "__main__":
    main()
