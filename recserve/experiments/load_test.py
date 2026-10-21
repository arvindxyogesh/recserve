"""M5: step load test with locust plus per-stage server timings from Prometheus.

For each target request rate the runner starts a headless locust run
(users = rate / per-user rate), then reads
  * client side: achieved req/s, p50/p95/p99, failures (locust stats CSV)
  * server side: per-stage p50/p95/p99 over the same window (Prometheus)
A step "sustains" its target when achieved >= 95% of target, p99 <= the
latency budget and the failure rate < 0.1%. Max sustained throughput is the
highest step that sustains.

Usage::

    python -m recserve.experiments.load_test --set load_test.label=baseline
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recserve.config import config_from_cli, run_tag
from recserve.utils import log, set_seed, setup_logging, write_result

STAGES = ("feature_fetch", "retrieval", "ranking", "total")
LOCUSTFILE = Path(__file__).with_name("locustfile.py")


def get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.load(resp)


def prom_quantiles(prom: str, window_s: int, at: float) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {s: {} for s in STAGES}
    for q in (0.5, 0.95, 0.99):
        expr = (
            f"histogram_quantile({q}, sum by (le, stage) "
            f"(rate(recserve_stage_seconds_bucket[{window_s}s])))"
        )
        url = f"{prom}/api/v1/query?" + urllib.parse.urlencode({"query": expr, "time": at})
        for r in get_json(url)["data"]["result"]:
            value = float(r["value"][1])
            out[r["metric"]["stage"]][f"p{int(q * 100)}"] = (
                round(value * 1000, 3) if math.isfinite(value) else None
            )
    return out


def docker_info() -> dict[str, Any]:
    """CPUs and memory of the Docker VM the API runs in, if docker is available."""
    try:
        raw = subprocess.check_output(
            ["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}} {{.ServerVersion}}"], text=True
        ).split()
        return {"cpus": int(raw[0]), "mem_gb": round(int(raw[1]) / 1e9, 1), "version": raw[2]}
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return {}


def run_step(rps: int, lcfg: dict[str, Any], env: dict[str, str], out_dir: Path) -> dict:
    users = max(1, math.ceil(rps / lcfg["per_user_rps"]))
    prefix = out_dir / f"rps{rps}"
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(LOCUSTFILE),
        "--headless",
        "--host",
        lcfg["host"],
        "-u",
        str(users),
        "-r",
        str(users),
        "-t",
        f"{lcfg['step_seconds']}s",
        "--csv",
        str(prefix),
        "--only-summary",
        "--processes",
        str(lcfg["locust_processes"]),
        "--stop-timeout",
        "5",
    ]
    t_start = time.time()
    # Hard timeout: under overload a multi-process locust run can fail to stop at -t.
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        proc.wait(timeout=lcfg["step_seconds"] + 60)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        timed_out = True
    stats_file = Path(f"{prefix}_stats.csv")
    if timed_out or not stats_file.exists():
        log.warning("step %d rps did not finish cleanly; marked invalid", rps)
        time.sleep(lcfg["cooldown_s"])
        return {"target_rps": rps, "locust_users": users, "valid": False}
    with open(stats_file) as f:
        agg = next(r for r in csv.DictReader(f) if r["Name"] == "Aggregated")
    n_req, n_fail = int(agg["Request Count"]), int(agg["Failure Count"])
    time.sleep(lcfg["prometheus_scrape_wait_s"])  # let the last scrape land
    # Server-side window = the planned step, never however long locust took to exit.
    window_end = t_start + lcfg["step_seconds"]
    window = lcfg["step_seconds"] - 5
    server = prom_quantiles(lcfg["prometheus"], window, window_end)
    time.sleep(lcfg["cooldown_s"])
    return {
        "target_rps": rps,
        "locust_users": users,
        "achieved_rps": round(float(agg["Requests/s"]), 1),
        "requests": n_req,
        "failure_rate": n_fail / max(n_req, 1),
        "client_ms": {"p50": float(agg["50%"]), "p95": float(agg["95%"]), "p99": float(agg["99%"])},
        "server_ms": server,
        "valid": True,
    }


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    rng = set_seed(cfg["seed"])
    lcfg = cfg["load_test"]
    tag = run_tag(cfg)
    server = get_json(f"{lcfg['host']}/info")
    log.info("server config %s", server)

    users = pd.read_parquet(Path(cfg["paths"]["split_dir"]) / "user_map.parquet")["userId"]
    work = Path(tempfile.mkdtemp(prefix="recserve_locust_"))
    ids_path = work / "user_ids.npy"
    np.save(ids_path, rng.choice(users.to_numpy(), size=lcfg["n_user_ids"], replace=False))
    env = {
        **os.environ,
        "RECSERVE_USER_IDS": str(ids_path),
        "RECSERVE_USER_RPS": str(lcfg["per_user_rps"]),
    }

    steps = []
    for rps in lcfg["target_rps"]:
        step = run_step(rps, lcfg, env, work)
        step["sustained"] = (
            step["valid"]
            and step["achieved_rps"] >= 0.95 * rps
            and step["client_ms"]["p99"] <= lcfg["p99_budget_ms"]
            and step["failure_rate"] < 0.001
        )
        log.info("%s", step)
        steps.append(step)
        if (
            lcfg["stop_after_failures"]
            and sum(not s["sustained"] for s in steps) >= lcfg["stop_after_failures"]
        ):
            break
    sustained = [s["target_rps"] for s in steps if s["sustained"]]
    best = max(sustained) if sustained else None

    label = lcfg["label"]
    header = (
        "| target rps | achieved rps | client p50 | p95 | p99 | fail % | server p99 total "
        "| fetch | retrieval | ranking | sustained |"
    )
    lines = [header, "|" + "---|" * 11]
    for s in steps:
        if not s["valid"]:
            lines.append(f"| {s['target_rps']} | invalid: locust did not stop |" + " |" * 9)
            continue
        sv = s["server_ms"]
        lines.append(
            f"| {s['target_rps']} | {s['achieved_rps']} | {s['client_ms']['p50']:.0f} "
            f"| {s['client_ms']['p95']:.0f} | {s['client_ms']['p99']:.0f} "
            f"| {100 * s['failure_rate']:.2f} | {sv['total'].get('p99')} "
            f"| {sv['feature_fetch'].get('p99')} | {sv['retrieval'].get('p99')} "
            f"| {sv['ranking'].get('p99')} | {'yes' if s['sustained'] else 'no'} |"
        )
    out = Path(cfg["paths"]["results_dir"])
    write_result(
        out / f"m5_load_{tag}_{label}.json",
        {
            "label": label,
            "server": server,
            "docker": docker_info(),
            "loadgen_cpus": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "load_test_config": lcfg,
            "max_sustained_rps": best,
            "steps": steps,
            "note": lcfg["environment_note"],
        },
        cfg,
    )
    (out / f"m5_load_{tag}_{label}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("max sustained rps:", best)


if __name__ == "__main__":
    main()
