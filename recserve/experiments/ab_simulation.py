"""M6: simulated A/B test, popularity (control) vs two-stage (treatment).

THIS IS AN OFFLINE SIMULATION, NOT A LIVE EXPERIMENT. Nobody saw these
recommendations. Held-out test-period ratings stand in for clicks, so the
result measures agreement with what users went on to rate, without position
bias, presentation effects or feedback loops.

Protocol (fixed before looking at test outcomes):
  1. Pre-registration on the val period only: estimate the control click
     rate and the per-arm sample size needed for the pre-registered MDE.
  2. Population: every user with at least one positive rating in the test
     period. Assignment: SHA-256 of salt + userId, even -> control, odd ->
     treatment. Sample-ratio-mismatch check with a chi-square test.
  3. Policies, both from the serving code (TwoStageRecommender):
       control   = 90-day popularity, excluding already-rated movies
       treatment = rec.serve(), i.e. exactly what GET /recommend returns:
                   two-stage for users with positive history, popularity
                   fallback for everyone else
  4. Click proxy: a user "clicks" if any of their top-10 recommendations is
     one of their held-out test positives. One binary outcome per user, so
     the unit of analysis is the unit of randomisation.
  5. Analysis: two-proportion z-test and Beta-Binomial posterior (vendored
     from ab-testing-framework), relative lift with CIs, pre-registered
     segments (warm = has positive history, cold = does not).
  6. Calibration: A/A re-splits of the control outcomes to check the false
     positive rate.

Runs where faiss and lightgbm share a process (the serving container, see
`make ab-test`).

Usage::

    python -m recserve.experiments.ab_simulation --config configs/default.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from recserve.config import config_from_cli, run_tag
from recserve.experiments.ab_stats import analyse, assign
from recserve.experiments.abtest_vendored import (
    power_curve,
    sample_size_proportion,
    two_proportion_z_test,
)
from recserve.serving.load_redis import build_user_state
from recserve.serving.recommender import TwoStageRecommender
from recserve.utils import log, set_seed, setup_logging, write_result

DAY_S = 86_400


def popularity_clicks(
    history: pd.DataFrame, period: pd.DataFrame, as_of: int, k: int, pool: int, window_days: int
) -> pd.DataFrame:
    """Control-policy clicks for every user with a positive in ``period``.

    Same rule as the serving cold-start list: top ``pool`` items by positives
    in the last ``window_days`` before ``as_of``, minus already-rated items,
    first k. Used for the val-period pre-registration estimate. Segment is
    "warm" for users with a positive in ``history``, as in serving.
    """
    pos = history[history["label"] == 1]
    counts = pos.loc[pos["ts"] >= as_of - window_days * DAY_S, "item_idx"].value_counts()
    top = counts.index.to_numpy()[:pool]
    seen = history.groupby("user_idx")["item_idx"].agg(set)
    warm = set(pos["user_idx"].unique())
    targets = period[period["label"] == 1].groupby("user_idx")["item_idx"].agg(set)
    rows = []
    for u, tgt in targets.items():
        s = seen.get(u, set())
        recs = [i for i in top if i not in s][:k]
        rows.append(
            {"segment": "warm" if u in warm else "cold", "click": bool(tgt.intersection(recs))}
        )
    return pd.DataFrame(rows)


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    rng = set_seed(cfg["seed"])
    acfg, scfg = cfg["ab_test"], cfg["serving"]
    k, alpha = acfg["k"], acfg["alpha"]
    tag = run_tag(cfg)
    state = build_user_state(cfg)
    parts, meta = state.parts, state.meta

    # 1. Pre-registration on the val period (no test data).
    val = popularity_clicks(
        parts["train"],
        parts["val"],
        meta["cutoffs"]["t_val"]["unix"],
        k,
        scfg["cold_start_pool"],
        cfg["baselines"]["popularity_recent_days"],
    )
    pre_reg = {}
    for seg, df in (
        ("overall", val),
        ("warm", val[val["segment"] == "warm"]),
        ("cold", val[val["segment"] == "cold"]),
    ):
        rate = float(df["click"].mean())
        plan = sample_size_proportion(rate, rate * acfg["mde_relative"], acfg["power"], alpha)
        pre_reg[seg] = {
            "control_rate": rate,
            "users": len(df),
            "mde_abs": rate * acfg["mde_relative"],
            "n_per_arm_required": plan.n_per_group,
        }
        log.info(
            "pre-registration %s: val control rate %.4f, need %d per arm for +%.0f%%",
            seg,
            rate,
            plan.n_per_group,
            100 * acfg["mde_relative"],
        )

    # 2. Population and assignment.
    test = parts["test"]
    targets = test[test["label"] == 1].groupby("user_idx")["item_idx"].agg(set)
    users = targets.index.to_numpy()
    user_ids = state.external[users]
    arm = assign(user_ids, acfg["salt"])
    n_t = int(arm.sum())
    srm = stats.chisquare([len(arm) - n_t, n_t])

    # 3-4. Outcomes for both policies (only the assigned one is analysed).
    rec = TwoStageRecommender(Path(cfg["paths"]["artifacts_dir"]) / "serving" / tag)
    rec.warmup()
    has_rating = np.diff(state.seen_start) > 0
    rows = []
    for u in users:
        record = state.record(u) if has_rating[u] else None
        tgt = targets[u]
        ctrl = rec.cold_start(k, None, None if record is None else record.seen)
        trt = rec.serve(record, k)
        rows.append(
            {
                "user_idx": u,
                "segment": "warm" if trt.source == "two_stage" else "cold",
                "click_control": bool(tgt.intersection(ctrl.items.tolist())),
                "click_treatment": bool(tgt.intersection(trt.items.tolist())),
                "short_control": len(ctrl.items) < k,
                "short_treatment": len(trt.items) < k,
            }
        )
    out = pd.DataFrame(rows)
    out["arm"] = arm
    out["click"] = np.where(arm == 1, out["click_treatment"], out["click_control"])
    art = Path(cfg["paths"]["artifacts_dir"]) / "ab" / tag
    art.mkdir(parents=True, exist_ok=True)
    out.to_parquet(art / "outcomes.parquet", index=False)

    # 5. Analysis, overall and by pre-registered segment.
    def by_arm(df: pd.DataFrame) -> dict:
        return analyse(
            df.loc[df["arm"] == 0, "click"].to_numpy(),
            df.loc[df["arm"] == 1, "click"].to_numpy(),
            alpha,
            cfg["seed"],
        )

    results = {"overall": by_arm(out)}
    for seg in ("warm", "cold"):
        results[seg] = by_arm(out[out["segment"] == seg])
    achieved_power = {}
    for seg in ("overall", "warm", "cold"):
        df = out if seg == "overall" else out[out["segment"] == seg]
        n_min = int(min((df["arm"] == 0).sum(), (df["arm"] == 1).sum()))
        _, pw = power_curve(
            pre_reg[seg]["control_rate"], n_min, np.array([pre_reg[seg]["mde_abs"]]), alpha
        )
        achieved_power[seg] = {"n_per_arm": n_min, "power_for_mde": float(pw[0])}

    # 6. A/A calibration on control-policy outcomes.
    ctrl_all = out["click_control"].to_numpy()
    aa_p = []
    for _ in range(acfg["aa_replications"]):
        mask = rng.permutation(len(ctrl_all)) < len(ctrl_all) // 2
        z = two_proportion_z_test(
            int((~mask).sum()),
            int(ctrl_all[~mask].sum()),
            int(mask.sum()),
            int(ctrl_all[mask].sum()),
            alpha=alpha,
        )
        aa_p.append(z.p_value)
    aa_p = np.array(aa_p)

    # Reference only: both policies for every user (impossible in a live test).
    paired = {
        seg: {
            "control_rate": float(df["click_control"].mean()),
            "treatment_rate": float(df["click_treatment"].mean()),
            "n": len(df),
        }
        for seg, df in (
            ("overall", out),
            ("warm", out[out["segment"] == "warm"]),
            ("cold", out[out["segment"] == "cold"]),
        )
    }

    payload: dict[str, Any] = {
        "disclaimer": "offline simulation on held-out ratings, not a live experiment",
        "click_definition": f"any of top-{k} recommendations is a held-out test positive",
        "pre_registration": {
            "period": "val",
            "mde_relative": acfg["mde_relative"],
            "power": acfg["power"],
            "alpha": alpha,
            "segments": pre_reg,
        },
        "population": {
            "users": int(len(users)),
            "control": int((arm == 0).sum()),
            "treatment": n_t,
            "srm_chi2": float(srm.statistic),
            "srm_p_value": float(srm.pvalue),
            "salt": acfg["salt"],
        },
        "achieved_power": achieved_power,
        "short_lists": {
            "control": int(out["short_control"].sum()),
            "treatment": int(out["short_treatment"].sum()),
        },
        "results": results,
        "aa_test": {
            "replications": int(len(aa_p)),
            "false_positive_rate": float((aa_p < alpha).mean()),
            "expected": alpha,
        },
        "paired_reference": paired,
        "serving": {"filter_mode": rec.filter_mode, "candidates": rec.n_candidates},
    }
    write_result(Path(cfg["paths"]["results_dir"]) / f"m6_ab_{tag}.json", payload, cfg)
    for seg, r in results.items():
        log.info(
            "%s: control %.4f treatment %.4f lift %+.1f%% CI [%+.1f%%, %+.1f%%] p=%.2g P(T>C)=%.4f",
            seg,
            r["control"]["rate"],
            r["treatment"]["rate"],
            100 * r["rel_lift"],
            100 * r["rel_lift_ci"][0],
            100 * r["rel_lift_ci"][1],
            r["p_value_precise"],
            r["bayes_prob_treatment_better"],
        )


if __name__ == "__main__":
    main()
