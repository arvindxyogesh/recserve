"""M3: LightGBM LambdaRank over two-tower candidates.

Data:
  * ranker training data = val period. Candidates come from the two-tower
    model trained on train only (artifacts/two_tower/<tag>/val), features
    from train-only history, labels = val positives. 20% of val users are
    held out for early stopping.
  * ranker evaluation = test period, with the train+val two-tower model and
    train+val history. The test set is used once, for the final numbers.

All rankings are scored against every test positive of the user, not only
the ones that made it into the candidate set, so ranking metrics include the
retrieval stage's misses.

Usage::

    python -m recserve.ranking.train_ranker --config configs/default.yaml
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from recserve.config import config_from_cli, run_tag
from recserve.data.history import RecentHistory
from recserve.data.item_features import ItemFeatures, build_item_features
from recserve.data.prepare import load_split
from recserve.eval.harness import EvalContext, top_k
from recserve.eval.metrics import (
    hits_matrix,
    metrics_from_hits,
    paired_bootstrap_ci,
    per_user_ranking,
)
from recserve.experiments.run_baselines import build_contexts
from recserve.ranking.features import (
    FEATURES,
    ITEM_STATS,
    PairFeaturizer,
    assemble,
    item_stats,
    user_stats,
)
from recserve.utils import log, set_seed, setup_logging, timed, write_result


@dataclass
class PeriodData:
    users: np.ndarray  # (U,)
    items: np.ndarray  # (U, N) candidates in retrieval order
    x: np.ndarray  # (U * N, F)
    y: np.ndarray  # (U * N,) 1 if the candidate is a positive in the period
    n_relevant: np.ndarray  # (U,) all positives in the period, retrieved or not


def retrieve(
    user_emb: np.ndarray, item_emb: np.ndarray, users: np.ndarray, ctx: EvalContext, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-n by inner product with already-rated items removed."""
    items, scores = [], []
    for s in range(0, len(users), 2048):
        batch = users[s : s + 2048]
        sc = user_emb[batch] @ item_emb.T
        sc[ctx.history_all[batch].nonzero()] = -np.inf
        top = top_k(sc, n)
        items.append(top)
        scores.append(np.take_along_axis(sc, top, axis=1))
    return np.concatenate(items), np.concatenate(scores)


def build_period(
    ctx: EvalContext, art: Path, feats: ItemFeatures, pair: PairFeaturizer, rcfg: dict[str, Any]
) -> tuple[PeriodData, np.ndarray]:
    users = ctx.users("warm")
    items, scores = retrieve(
        np.load(art / "user_emb.npy"),
        np.load(art / "item_emb.npy"),
        users,
        ctx,
        rcfg["candidates"],
    )
    n = items.shape[1]
    ustats = user_stats(ctx.history, ctx.n_users, ctx.start_ts)
    istats = item_stats(ctx.history, ctx.n_items, ctx.start_ts, feats)
    recent = RecentHistory.from_df(ctx.history, ctx.n_users).matrix(users, rcfg["history_len"])
    x = assemble(
        np.repeat(ustats[users], n, axis=0), istats[items.ravel()], pair(recent, items, scores)
    )
    y = hits_matrix(items, users, ctx.target_pos).ravel().astype(np.float32)
    n_rel = np.diff(ctx.target_pos.indptr)[users]
    return PeriodData(users, items, x, y, n_rel), istats


def lgb_dataset(d: PeriodData, user_mask: np.ndarray, ref: lgb.Dataset | None = None):
    """Rows for the selected users; users with no positive candidate carry no signal."""
    n = d.items.shape[1]
    has_pos = d.y.reshape(-1, n).any(axis=1)
    keep = user_mask & has_pos
    rows = np.repeat(keep, n)
    return lgb.Dataset(
        d.x[rows],
        d.y[rows],
        group=np.full(keep.sum(), n),
        feature_name=FEATURES,
        reference=ref,
        free_raw_data=False,
    ), int(keep.sum())


def rank_metrics(d: PeriodData, order_scores: np.ndarray, ks: list[int], k: int):
    """Reorder each user's candidates by order_scores (U, N) and score the ranked list."""
    order = np.argsort(-order_scores, axis=1, kind="stable")
    hits = np.take_along_axis(d.y.reshape(order_scores.shape).astype(bool), order, axis=1)
    return metrics_from_hits(hits, d.n_relevant, ks, k), per_user_ranking(hits, d.n_relevant, k)


def predict_latency_ms(booster: lgb.Booster, d: PeriodData, n_users: int, seed: int):
    """Single-threaded predict for one request (one user's candidates), as in serving."""
    n = d.items.shape[1]
    rng = np.random.default_rng(seed)
    times = []
    for u in rng.choice(len(d.users), size=min(n_users, len(d.users)), replace=False):
        x = d.x[u * n : (u + 1) * n]
        t0 = time.perf_counter()
        booster.predict(x, num_threads=1)
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {f"p{q}": round(float(np.percentile(t, q)), 4) for q in (50, 95, 99)}


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    rng = set_seed(cfg["seed"])
    rcfg, ecfg = cfg["ranker"], cfg["eval"]
    tag = run_tag(cfg)
    split_dir = Path(cfg["paths"]["split_dir"])
    tt_dir = Path(cfg["paths"]["artifacts_dir"]) / "two_tower" / tag
    parts, meta = load_split(split_dir)
    item_map = pd.read_parquet(split_dir / "item_map.parquet")
    feats = build_item_features(cfg["paths"]["processed_dir"], item_map, use_genome=True)
    pair = PairFeaturizer(feats)
    val_ctx, test_ctx = build_contexts(parts, meta)

    timings: dict[str, float] = {}
    with timed("features val", timings):
        val, _ = build_period(val_ctx, tt_dir / "val", feats, pair, rcfg)
    with timed("features test", timings):
        test, test_istats = build_period(test_ctx, tt_dir / "test", feats, pair, rcfg)
    n = test.items.shape[1]

    holdout = np.zeros(len(val.users), dtype=bool)
    holdout[rng.choice(len(val.users), int(rcfg["valid_frac"] * len(val.users)), False)] = True
    dtrain, n_train = lgb_dataset(val, ~holdout)
    dvalid, n_valid = lgb_dataset(val, holdout, ref=dtrain)
    params = {**rcfg["params"], "seed": cfg["seed"]}
    log.info("ranker train users %d, early-stopping users %d", n_train, n_valid)
    with timed("train ranker", timings):
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=rcfg["num_boost_round"],
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(rcfg["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(100),
            ],
        )
    log.info("best iteration %d", booster.best_iteration)

    ks, k = [10, 50, n], ecfg["ranking_k"]
    test_scores = booster.predict(test.x, num_iteration=booster.best_iteration).reshape(-1, n)
    retrieval_scores = test.x[:, FEATURES.index("retrieval_score")].reshape(-1, n)
    pop_scores = test_istats[test.items, ITEM_STATS.index("i_pop_90d")]
    orders = {
        "two_tower (retrieval order)": retrieval_scores,
        "two_tower + popularity_90d rerank": pop_scores,
        "two_tower + lambdarank": test_scores,
    }
    rows, per_user = [], {}
    for name, sc in orders.items():
        m, pu = rank_metrics(test, sc, ks, k)
        rows.append({"model": name, "n_users": len(test.users), "metrics": m})
        per_user[name] = pu
        log.info("%s %s", name, m)

    pop_name = f"popularity_{cfg['baselines']['popularity_recent_days']}d"
    m1 = json.loads((Path(cfg["paths"]["results_dir"]) / f"m1_baselines_{tag}.json").read_text())
    for r in m1["results"]:
        if (
            r["period"] == "test"
            and r["users_kind"] == "warm"
            and r["model"] in (pop_name, "cooccurrence")
        ):
            rows.append(
                {
                    "model": f"{r['model']} (full catalogue, M1)",
                    "n_users": r["n_users"],
                    "metrics": r["metrics"],
                }
            )

    ranker = per_user["two_tower + lambdarank"]
    lifts = {
        "ndcg@10 vs retrieval order": paired_bootstrap_ci(
            ranker["ndcg"], per_user["two_tower (retrieval order)"]["ndcg"], seed=cfg["seed"]
        ),
        "ndcg@10 vs popularity_90d rerank": paired_bootstrap_ci(
            ranker["ndcg"],
            per_user["two_tower + popularity_90d rerank"]["ndcg"],
            seed=cfg["seed"],
        ),
    }
    pop_full = next(r for r in rows if r["model"].startswith(f"{pop_name} (full"))
    lifts[f"ndcg@10 vs {pop_name} full catalogue (point estimate)"] = {
        "lift": rows[2]["metrics"]["ndcg@10"] / pop_full["metrics"]["ndcg@10"] - 1
    }

    gain = booster.feature_importance("gain", iteration=booster.best_iteration)
    split = booster.feature_importance("split", iteration=booster.best_iteration)
    importance = sorted(
        (
            {"feature": f, "gain_share": float(g / gain.sum()), "splits": int(s)}
            for f, g, s in zip(FEATURES, gain, split, strict=True)
        ),
        key=lambda r: -r["gain_share"],
    )
    latency = predict_latency_ms(booster, test, 1000, cfg["seed"])

    art = Path(cfg["paths"]["artifacts_dir"]) / "ranker" / tag
    art.mkdir(parents=True, exist_ok=True)
    booster.save_model(art / "model.txt", num_iteration=booster.best_iteration)
    (art / "features.json").write_text(json.dumps(FEATURES) + "\n")

    cols = [f"ndcg@{k}", f"mrr@{k}", f"hitrate@{k}", f"recall@{k}", f"recall@{n}"]
    table = ["| model | users | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for r in rows:
        vals = " | ".join(f"{r['metrics'][c]:.4f}" if c in r["metrics"] else "n/a" for c in cols)
        table.append(f"| {r['model']} | {r['n_users']} | {vals} |")
    out = Path(cfg["paths"]["results_dir"])
    write_result(
        out / f"m3_ranker_{tag}.json",
        {
            "candidates_per_user": n,
            "ranker_train_users": n_train,
            "ranker_early_stopping_users": n_valid,
            "val_candidate_positive_rate": float(val.y.mean()),
            "best_iteration": booster.best_iteration,
            "lightgbm": lgb.__version__,
            "results": rows,
            "lifts": lifts,
            "feature_importance": importance,
            "predict_latency_ms_per_request": latency,
            "timings_s": timings,
        },
        cfg,
    )
    (out / f"m3_ranker_{tag}.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(json.dumps(lifts, indent=1))


if __name__ == "__main__":
    main()
