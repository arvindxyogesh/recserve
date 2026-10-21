"""Seeding, timing, hardware description and result writing shared by all scripts."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from recserve.config import config_hash

log = logging.getLogger("recserve")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    # Seed torch only if this process already uses it. Importing it here would
    # load its OpenMP runtime next to faiss's, which aborts on macOS.
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
    return np.random.default_rng(seed)


@contextmanager
def timed(name: str, sink: dict[str, float] | None = None) -> Iterator[None]:
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if sink is not None:
        sink[name] = round(elapsed, 3)
    log.info("%s took %.2fs", name, elapsed)


def hardware_info() -> dict[str, Any]:
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        with suppress(OSError, subprocess.CalledProcessError):
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        ram_gb = None
    return {
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_result(path: str | Path, payload: dict[str, Any], cfg: dict[str, Any]) -> Path:
    """Write a result JSON with the provenance needed to reproduce it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config_path": cfg.get("_config_path"),
        "config_hash": config_hash(cfg),
        "seed": cfg.get("seed"),
        "hardware": hardware_info(),
        **payload,
    }
    path.write_text(json.dumps(record, indent=2, default=_json_default) + "\n")
    log.info("wrote %s", path)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")
