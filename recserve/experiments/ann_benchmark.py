"""M2: HNSW vs exact search on the exported two-tower embeddings (test-period model).

Run on the serving machine: latency is per single query on one thread, the
way one request is served. Reports build time, index size, latency
percentiles, overlap with exact top-k, and end-to-end Recall@k on the test
period after filtering already-seen items.

Usage::

    python -m recserve.experiments.ann_benchmark --config configs/default.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np

from recserve.config import config_from_cli, run_tag
from recserve.data.prepare import load_split
from recserve.eval.metrics import hits_matrix
from recserve.experiments.run_baselines import build_contexts
from recserve.retrieval.ann import (
    build_index,
    index_bytes,
    latency_ms,
    search_filtered,
    set_ef_search,
)
from recserve.utils import log, set_seed, setup_logging, write_result


def overlap_at_k(a: np.ndarray, b: np.ndarray) -> float:
    k = a.shape[1]
    return float(np.mean([len(np.intersect1d(x, y)) / k for x, y in zip(a, b, strict=True)]))


def recall_from_ids(ids: np.ndarray, users: np.ndarray, target) -> float:
    hits = hits_matrix(np.where(ids < 0, 0, ids), users, target) & (ids >= 0)
    n_rel = np.diff(target.indptr)[users]
    return float(np.mean(hits.sum(axis=1) / n_rel))


def pct(x: np.ndarray) -> dict[str, float]:
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in (50, 95, 99)} | {
        "mean": round(float(x.mean()), 4)
    }


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    rng = set_seed(cfg["seed"])
    acfg = cfg["ann"]
    tag = run_tag(cfg)
    art = Path(cfg["paths"]["artifacts_dir"]) / "two_tower" / tag / "test"
    item_emb = np.load(art / "item_emb.npy")
    user_emb = np.load(art / "user_emb.npy")
    parts, meta = load_split(cfg["paths"]["split_dir"])
    _, ctx = build_contexts(parts, meta)

    users = ctx.users("warm")
    queries = user_emb[users]
    seen = [
        ctx.history_all.indices[ctx.history_all.indptr[u] : ctx.history_all.indptr[u + 1]]
        for u in users
    ]
    lat_users = rng.choice(len(users), size=min(acfg["latency_queries"], len(users)), replace=False)
    k = acfg["k"]
    faiss.omp_set_num_threads(acfg["threads"])

    flat, flat_build = build_index(item_emb, "flat")
    exact_ids = search_filtered(flat, queries, seen, k)
    _, exact_raw = flat.search(queries[lat_users], k)
    flat_lat = latency_ms(flat, queries[lat_users], k)
    rows: list[dict[str, Any]] = [
        {
            "index": "flat (exact)",
            "build_s": round(flat_build, 3),
            "index_mb": round(index_bytes(flat) / 1e6, 2),
            "latency_ms": pct(flat_lat),
            "overlap_with_exact": 1.0,
            f"recall@{k}_test": recall_from_ids(exact_ids, users, ctx.target_pos),
        }
    ]
    log.info("%s", rows[-1])

    for m in acfg["hnsw_m"]:
        index, build_s = build_index(item_emb, "hnsw", m, acfg["ef_construction"])
        size_mb = round(index_bytes(index) / 1e6, 2)
        for ef in acfg["ef_search"]:
            set_ef_search(index, ef)
            _, raw = index.search(queries[lat_users], k)
            lat = latency_ms(index, queries[lat_users], k)
            ids = search_filtered(index, queries, seen, k, ef=ef)
            rows.append(
                {
                    "index": f"hnsw M={m} efSearch={ef}",
                    "m": m,
                    "ef_search": ef,
                    "ef_construction": acfg["ef_construction"],
                    "build_s": round(build_s, 3),
                    "index_mb": size_mb,
                    "latency_ms": pct(lat),
                    "overlap_with_exact": overlap_at_k(raw, exact_raw),
                    f"recall@{k}_test": recall_from_ids(ids, users, ctx.target_pos),
                }
            )
            log.info("%s", rows[-1])

    short = sum(1 for s in seen if k + len(s) > 2048)
    table = [
        f"| index | build s | size MB | p50 ms | p95 ms | p99 ms | overlap@{k} | "
        f"Recall@{k} (test) |",
        "|---|---|---|---|---|---|---|---|",
    ] + [
        f"| {r['index']} | {r['build_s']} | {r['index_mb']} | {r['latency_ms']['p50']:.3f} "
        f"| {r['latency_ms']['p95']:.3f} | {r['latency_ms']['p99']:.3f} "
        f"| {r['overlap_with_exact']:.4f} | {r[f'recall@{k}_test']:.4f} |"
        for r in rows
    ]
    out_dir = Path(cfg["paths"]["results_dir"])
    write_result(
        out_dir / f"m2_ann_{tag}.json",
        {
            "n_items": int(item_emb.shape[0]),
            "dim": int(item_emb.shape[1]),
            "n_eval_users": int(len(users)),
            "n_latency_queries": int(len(lat_users)),
            "users_with_truncated_search": short,
            "faiss": faiss.__version__,
            "ann_config": acfg,
            "results": rows,
        },
        cfg,
    )
    (out_dir / f"m2_ann_{tag}.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
