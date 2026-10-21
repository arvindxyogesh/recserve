"""M7: headline results table, read from results/*.json (no new measurements).

Usage::

    python -m recserve.experiments.report --config configs/default.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

from recserve.config import config_from_cli, run_tag


def load(res: Path, name: str) -> dict:
    return json.loads((res / name).read_text())


def main() -> None:
    cfg = config_from_cli(__doc__.splitlines()[0])
    tag = run_tag(cfg)
    res = Path(cfg["paths"]["results_dir"])
    m1 = load(res, f"m1_baselines_{tag}.json")
    m2 = load(res, f"m2_two_tower_{tag}.json")
    ann = load(res, f"m2_ann_{tag}.json")
    m3 = load(res, f"m3_ranker_{tag}.json")
    probe = load(res, f"m4_serving_probe_{tag}.json")
    m5b = load(res, f"m5_load_{tag}_baseline.json")
    m5s = load(res, f"m5_load_{tag}_selector.json")
    q = {v: load(res, f"m5_quality_{tag}_{v}.json") for v in ("overfetch_c100", "selector_c100")}
    m6 = load(res, f"m6_ab_{tag}.json")

    lines = [
        "# recserve results",
        "",
        "All numbers are read from results/*.json by recserve/experiments/report.py.",
        "",
        "## Retrieval and ranking (test period, warm users)",
        "",
        "| model | users | Recall@100 | NDCG@10 | HitRate@10 |",
        "|---|---|---|---|---|",
    ]
    for r in m1["results"]:
        if r["period"] == "test" and r["users_kind"] == "warm" and r["model"] != "random":
            m = r["metrics"]
            lines.append(
                f"| {r['model']} | {r['n_users']} | {m['recall@100']:.4f} "
                f"| {m['ndcg@10']:.4f} | {m['hitrate@10']:.4f} |"
            )
    t = m2["test"]["result"]
    lines.append(
        f"| two_tower | {t['n_users']} | {t['metrics']['recall@100']:.4f} "
        f"| {t['metrics']['ndcg@10']:.4f} | {t['metrics']['hitrate@10']:.4f} |"
    )
    for r in m3["results"]:
        if r["model"] == "two_tower + lambdarank":
            m = r["metrics"]
            lines.append(
                f"| two_tower + lambdarank | {r['n_users']} | {m['recall@100']:.4f} "
                f"| {m['ndcg@10']:.4f} | {m['hitrate@10']:.4f} |"
            )
    lift = m3["lifts"]["ndcg@10 vs retrieval order"]
    lines += [
        "",
        f"LambdaRank NDCG@10 lift over retrieval order: {100 * lift['lift']:+.1f}% "
        f"(95% paired bootstrap CI {100 * lift['lift_ci'][0]:+.1f}% to "
        f"{100 * lift['lift_ci'][1]:+.1f}%).",
    ]
    pop_key = next(k for k in m3["lifts"] if k.startswith("ndcg@10 vs popularity") and "full" in k)
    lines.append(
        f"LambdaRank NDCG@10 lift over 90-day popularity (full catalogue): "
        f"{100 * m3['lifts'][pop_key]['lift']:+.1f}%."
    )

    lines += [
        "",
        "## ANN index (62,423 items, 64-d, 1 thread, k=100)",
        "",
        "| index | build s | size MB | p99 ms | overlap with exact |",
        "|---|---|---|---|---|",
    ]
    for r in ann["results"]:
        if r["index"] in ("flat (exact)", "hnsw M=16 efSearch=128"):
            lines.append(
                f"| {r['index']} | {r['build_s']} | {r['index_mb']} "
                f"| {r['latency_ms']['p99']:.3f} | {r['overlap_with_exact']:.4f} |"
            )

    p = probe["two_stage_latency_ms"]
    lines += [
        "",
        "## Serving latency, sequential single client (ms)",
        "",
        "| stage | p50 | p99 |",
        "|---|---|---|",
    ]
    lines += [
        f"| {s} | {p[s]['p50']} | {p[s]['p99']} |"
        for s in ("feature_fetch", "retrieval", "ranking", "total", "client_rtt")
    ]

    def at(run: dict, rps: int) -> dict:
        return next(s for s in run["steps"] if s["target_rps"] == rps)

    best_b, best_s = m5b["max_sustained_rps"], m5s["max_sustained_rps"]
    lines += [
        "",
        "## Load test (locust, 4 API workers on 4 pinned vCPUs)",
        "",
        "| variant | max sustained req/s | client p99 at that rate (ms) "
        "| server retrieval p99 at that rate (ms) | NDCG@10 (serving path) "
        "| in-process mean ms |",
        "|---|---|---|---|---|---|",
    ]
    for name, run, best, qq in (
        ("over-fetch (before)", m5b, best_b, q["overfetch_c100"]),
        ("FAISS ID selector (after)", m5s, best_s, q["selector_c100"]),
    ):
        st = at(run, best)
        lines.append(
            f"| {name} | {best} | {st['client_ms']['p99']:.0f} "
            f"| {st['server_ms']['retrieval']['p99']} | {qq['ranked']['ndcg@10']:.4f} "
            f"| {qq['in_process_ms']['total']['mean']:.3f} |"
        )

    lines += [
        "",
        "## Simulated A/B test (offline replay, not a live experiment)",
        "",
        "| segment | control | treatment | relative lift (95% CI) | p-value |",
        "|---|---|---|---|---|",
    ]
    for seg, r in m6["results"].items():
        lines.append(
            f"| {seg} | {100 * r['control']['rate']:.1f}% (n={r['control']['n']}) "
            f"| {100 * r['treatment']['rate']:.1f}% (n={r['treatment']['n']}) "
            f"| {100 * r['rel_lift']:+.1f}% ({100 * r['rel_lift_ci'][0]:+.1f}% to "
            f"{100 * r['rel_lift_ci'][1]:+.1f}%) | {r['p_value_precise']:.2g} |"
        )
    lines += [
        "",
        f"SRM p = {m6['population']['srm_p_value']:.2f}; A/A false positive rate "
        f"{100 * m6['aa_test']['false_positive_rate']:.1f}% over "
        f"{m6['aa_test']['replications']} re-splits (alpha 5%).",
    ]

    out = res / f"summary_{tag}.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
