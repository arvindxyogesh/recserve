"""YAML config loading with dotted-key overrides (e.g. ``--set data.sample_users=5000``)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    cfg["_config_path"] = str(path)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    clean = {k: v for k, v in copy.deepcopy(cfg).items() if not k.startswith("_")}
    return hashlib.sha1(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:10]


def run_tag(cfg: dict[str, Any]) -> str:
    """Short name for result files, e.g. ``ml-25m`` or ``ml-25m_sample5000``."""
    tag = cfg["data"]["dataset"]
    if cfg["data"].get("sample_users") is not None:
        tag += f"_sample{cfg['data']['sample_users']}"
    return tag


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set data.sample_users=5000",
    )
    return parser


def config_from_cli(description: str) -> dict[str, Any]:
    args = add_config_args(argparse.ArgumentParser(description=description)).parse_args()
    return load_config(args.config, args.overrides)
