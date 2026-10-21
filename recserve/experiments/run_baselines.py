"""M1: evaluate random, popularity and co-occurrence baselines on val and test.

For each period the models are refit on everything before it (see
recserve/eval/harness.py). Warm users get every model; cold users are
evaluated with popularity only, since the other models need history.

Usage::

    python -m recserve.experiments.run_baselines --config configs/default.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from recserve.config import config_from_cli, run_tag
from recserve.data.prepare import load_split
from recserve.eval.harness import EvalContext, evaluate
from recserve.retrieval.baselines import (
    CooccurrenceRecommender,
    PopularityRecommender,
    RandomRecommender,
)
from recserve.utils import log, set_seed, setup_logging, write_result


def build_contexts(parts: dict[str, pd.DataFrame], meta: dict[str, Any]) -> list[EvalContext]:
    n_users, n_items = meta["n_users"], meta["n_items"]
    val_ctx = EvalContext(
        "val", parts["train"], parts["val"], n_users, n_items, meta["cutoffs"]["t_val"]["unix"]
    )
    test_ctx = EvalContext(
        "test",
        pd.concat([parts["train"], parts["val"]], ignore_index=True),
        parts["test"],
        n_users,
        n_items,
        meta["cutoffs"]["t_test"]["unix"],
    )
    return [val_ctx, test_ctx]


def results_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "| period | users | model | " + " | ".join(columns) + " |"
    sep = "|" + "---|" * (len(columns) + 3)
    lines = [head, sep]
    for r in rows:
        vals = " | ".join(f"{r['metrics'][c]:.4f}" for c in columns)
        lines.append(
            f"| {r['period']} | {r['users_kind']} ({r['n_users']}) | {r['model']} | {vals} |"
        )
    return "\n".join(lines)


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    rng = set_seed(cfg["seed"])
    ecfg, bcfg = cfg["eval"], cfg["baselines"]
    parts, meta = load_split(cfg["paths"]["split_dir"])

    models = [
        RandomRecommender(seed=cfg["seed"]),
        PopularityRecommender(),
        PopularityRecommender(window_days=bcfg["popularity_recent_days"]),
        CooccurrenceRecommender(**bcfg["cooccurrence"]),
    ]
    common = dict(
        retrieval_ks=ecfg["retrieval_ks"],
        ranking_k=ecfg["ranking_k"],
        batch_size=ecfg["batch_size"],
        max_users=ecfg["max_eval_users"],
        rng=rng,
    )

    rows: list[dict[str, Any]] = []
    for ctx in build_contexts(parts, meta):
        for model in models:
            log.info("evaluating %s on %s (warm users)", model.name, ctx.name)
            res = evaluate(model, ctx, **common)
            rows.append({**res, "users_kind": "warm"})
            log.info("%s", res)
            if isinstance(model, PopularityRecommender):
                res = evaluate(model, ctx, users=ctx.users("cold"), fit=False, **common)
                rows.append({**res, "users_kind": "cold"})
                log.info("%s", res)

    k = ecfg["ranking_k"]
    columns = [f"recall@{x}" for x in ecfg["retrieval_ks"]] + [
        f"ndcg@{k}",
        f"mrr@{k}",
        f"hitrate@{k}",
    ]
    table = results_table(rows, columns)
    tag = run_tag(cfg)
    out_dir = Path(cfg["paths"]["results_dir"])
    write_result(
        out_dir / f"m1_baselines_{tag}.json",
        {
            "dataset": {k_: meta[k_] for k_ in ("n_users", "n_items", "sample_users")}
            | {"train_ratings": meta["train"]["ratings"], "cutoffs": meta["cutoffs"]},
            "eval_config": ecfg,
            "baseline_config": bcfg,
            "results": rows,
        },
        cfg,
    )
    (out_dir / f"m1_baselines_{tag}.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
