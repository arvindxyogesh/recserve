"""Export the user tower of a trained TwoTower to a numpy .npz for serving.

Usage::

    python -m recserve.retrieval.export_user_tower --config configs/default.yaml
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recserve.config import config_from_cli, run_tag
from recserve.data.item_features import build_item_features
from recserve.retrieval.two_tower import TwoTower
from recserve.utils import log, setup_logging


def export_user_tower(model: TwoTower, path: Path) -> None:
    lin0, lin1 = model.user_mlp[0], model.user_mlp[3]
    to_np = lambda t: t.detach().float().cpu().numpy()  # noqa: E731
    np.savez(
        path,
        item_id=to_np(model.item_id.weight),
        genres=to_np(model.genres),
        w0=to_np(lin0.weight),
        b0=to_np(lin0.bias),
        w1=to_np(lin1.weight),
        b1=to_np(lin1.bias),
    )
    log.info("wrote %s", path)


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    tcfg = cfg["two_tower"]
    item_map = pd.read_parquet(Path(cfg["paths"]["split_dir"]) / "item_map.parquet")
    feats = build_item_features(cfg["paths"]["processed_dir"], item_map, tcfg["use_genome"])
    for period in ("val", "test"):
        art = Path(cfg["paths"]["artifacts_dir"]) / "two_tower" / run_tag(cfg) / period
        model = TwoTower(feats, tcfg["emb_dim"], tcfg["hidden"], tcfg["dropout"])
        model.load_state_dict(torch.load(art / "model.pt", map_location="cpu"))
        export_user_tower(model.eval(), art / "user_tower.npz")


if __name__ == "__main__":
    main()
