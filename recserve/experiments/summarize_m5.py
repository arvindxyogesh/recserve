"""Build results/m5_summary_<tag>.md from the M5 result JSONs (no new measurements).

Usage::

    python -m recserve.experiments.summarize_m5 --config configs/default.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

from recserve.config import config_from_cli, run_tag
from recserve.experiments.load_test import docker_info
from recserve.utils import hardware_info


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    cfg = config_from_cli(__doc__.splitlines()[0])
    tag = run_tag(cfg)
    res = Path(cfg["paths"]["results_dir"])
    out = ["# M5 summary", ""]

    out += [
        "## Serving-path quality and in-process cost (all warm test users, 1 thread)",
        "",
        "| filter | candidates | NDCG@10 | Recall@candidates | retrieval p50 / p99 ms "
        "| ranking p50 / p99 ms | total mean / p99 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in sorted(res.glob(f"m5_quality_{tag}_*.json")):
        d = load(f)
        ip, v = d["in_process_ms"], d["variant"]
        out.append(
            f"| {v['filter_mode']} | {v['candidates']} | {d['ranked']['ndcg@10']:.4f} "
            f"| {next(iter(d['retrieval'].values())):.4f} "
            f"| {ip['retrieval']['p50']:.3f} / {ip['retrieval']['p99']:.3f} "
            f"| {ip['ranking']['p50']:.3f} / {ip['ranking']['p99']:.3f} "
            f"| {ip['total']['mean']:.3f} / {ip['total']['p99']:.3f} |"
        )

    runs = {lbl: load(res / f"m5_load_{tag}_{lbl}.json") for lbl in ("baseline", "selector")}
    out += [
        "",
        "## Load test: before (over-fetch) vs after (FAISS ID selector)",
        "",
        "| target rps | before: achieved / client p99 / server retrieval p99 "
        "| after: achieved / client p99 / server retrieval p99 |",
        "|---|---|---|",
    ]
    after = {s["target_rps"]: s for s in runs["selector"]["steps"]}
    for s in runs["baseline"]["steps"]:
        a = after.get(s["target_rps"])

        def cell(x: dict | None) -> str:
            if not x or not x.get("valid", True):
                return "n/a"
            mark = "" if x["sustained"] else " (not sustained)"
            return (
                f"{x['achieved_rps']} / {x['client_ms']['p99']:.0f} ms / "
                f"{x['server_ms']['retrieval'].get('p99')} ms{mark}"
            )

        out.append(f"| {s['target_rps']} | {cell(s)} | {cell(a)} |")
    out += [
        "",
        "Max sustained throughput (achieved >= 95% of target, client p99 <= "
        f"{runs['baseline']['load_test_config']['p99_budget_ms']} ms, failures < 0.1%): "
        f"before {runs['baseline']['max_sustained_rps']} req/s, "
        f"after {runs['selector']['max_sustained_rps']} req/s.",
        "",
        f"Environment: {runs['baseline']['note']}. API: {runs['baseline']['server']}. "
        f"Docker VM (read on the host when summarising): {docker_info() or 'n/a'}. "
        f"Host: {hardware_info()}.",
    ]
    path = res / f"m5_summary_{tag}.md"
    path.write_text("\n".join(out) + "\n")
    print(path.read_text())


if __name__ == "__main__":
    main()
