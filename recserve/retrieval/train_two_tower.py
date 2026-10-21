"""M2: train the two-tower model, evaluate it with the shared harness, export embeddings.

Same protocol as the M1 baselines: for each period the model is trained on
everything before it. The val model picks the number of epochs (best
Recall@100 on a sample of val users); the test model is then trained on
train + val for exactly that many epochs, so the test set never influences a
choice.

Usage::

    python -m recserve.retrieval.train_two_tower --config configs/default.yaml
    # throughput check only (no eval or export):
    python -m recserve.retrieval.train_two_tower --set two_tower.benchmark_steps=300
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from recserve.config import config_from_cli, run_tag
from recserve.data.item_features import ItemFeatures, build_item_features
from recserve.data.prepare import load_split
from recserve.eval.harness import EvalContext, evaluate
from recserve.experiments.run_baselines import build_contexts
from recserve.retrieval.two_tower import (
    Sequences,
    TwoTower,
    history_matrix,
    in_batch_softmax_loss,
)
from recserve.utils import log, set_seed, setup_logging, write_result


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type


class TwoTowerRecommender:
    """Adapter from a trained TwoTower to the harness Recommender protocol."""

    def __init__(self, model: TwoTower, history_len: int, device: torch.device) -> None:
        self.name = "two_tower"
        self.model = model
        self.history_len = history_len
        self.device = device

    @torch.no_grad()
    def fit(self, ctx: EvalContext) -> None:
        self.model.eval()
        all_items = torch.arange(ctx.n_items, device=self.device)
        self.item_emb = torch.cat(
            [self.model.item_embed(chunk) for chunk in all_items.split(16_384)]
        )
        seq = Sequences.from_positives(ctx.history[ctx.history["label"] == 1], ctx.n_users)
        self.items = torch.from_numpy(seq.items).to(self.device)
        self.user_start = torch.from_numpy(seq.user_start).to(self.device)

    @torch.no_grad()
    def user_embeddings(self, users: np.ndarray) -> torch.Tensor:
        u = torch.from_numpy(users.astype(np.int64)).to(self.device)
        hist = history_matrix(
            self.items,
            self.user_start[u],
            self.user_start[u + 1],
            self.history_len,
            self.model.pad,
        )
        return self.model.user_embed(hist)

    def score(self, users: np.ndarray, ctx: EvalContext) -> np.ndarray:
        scores = self.user_embeddings(users) @ self.item_emb.T
        return scores.float().cpu().numpy()


def train(
    feats: ItemFeatures,
    ctx: EvalContext,
    tcfg: dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
    epochs: int,
    eval_ctx: EvalContext | None = None,
    eval_kwargs: dict[str, Any] | None = None,
) -> tuple[TwoTower, list[dict[str, Any]], int]:
    """Train on ctx.history positives; if eval_ctx is given, keep the best epoch on it."""
    model = TwoTower(feats, tcfg["emb_dim"], tcfg["hidden"], tcfg["dropout"]).to(device)
    seq = Sequences.from_positives(ctx.history[ctx.history["label"] == 1], ctx.n_users)
    targets = seq.targets()
    items_t = torch.from_numpy(seq.items).to(device)
    start_t = torch.from_numpy(seq.start_of).to(device)
    counts = np.bincount(seq.items[targets], minlength=feats.n_items).astype(np.float64)
    log_q = torch.from_numpy(np.log(np.maximum(counts, 1.0) / counts.sum())).float().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    bs, hl, tau = tcfg["batch_size"], tcfg["history_len"], tcfg["temperature"]
    max_steps = tcfg.get("benchmark_steps")
    log.info(
        "training on %d examples (%d users), %d epochs, device %s",
        len(targets),
        int((np.diff(seq.user_start) > 1).sum()),
        epochs,
        device,
    )

    curve: list[dict[str, Any]] = []
    best = (-1.0, 0, None)
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(targets)
        t0, loss_sum, n_batches = time.perf_counter(), 0.0, 0
        for b in range(0, len(perm), bs):
            pos = torch.from_numpy(perm[b : b + bs]).to(device)
            tgt = items_t[pos]
            hist = history_matrix(items_t, start_t[pos], pos, hl, model.pad)
            loss = in_batch_softmax_loss(
                model.user_embed(hist),
                model.item_embed(tgt),
                tgt,
                tau,
                log_q[tgt] if tcfg["logq_correction"] else None,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            n_batches += 1
            step += 1
            if step % 500 == 0:
                log.info(
                    "epoch %d step %d loss %.4f %.1f steps/s",
                    epoch,
                    step,
                    loss_sum / n_batches,
                    n_batches / (time.perf_counter() - t0),
                )
            if max_steps and step >= max_steps:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        row: dict[str, Any] = {
            "epoch": epoch,
            "loss": loss_sum / n_batches,
            "steps": n_batches,
            "epoch_s": round(elapsed, 2),
            "steps_per_s": round(n_batches / elapsed, 2),
        }
        if max_steps:
            curve.append(row)
            return model, curve, epoch
        if eval_ctx is not None:
            res = evaluate(TwoTowerRecommender(model, hl, device), eval_ctx, **(eval_kwargs or {}))
            row["sample_eval"] = res["metrics"]
            score = res["metrics"]["recall@100"]
            if score > best[0]:
                best = (score, epoch, copy.deepcopy(model.state_dict()))
        log.info("%s", row)
        curve.append(row)

    best_epoch = epochs
    if eval_ctx is not None:
        best_epoch = best[1]
        model.load_state_dict(best[2])
    return model, curve, best_epoch


def export(
    model: TwoTower,
    ctx: EvalContext,
    history_len: int,
    device: torch.device,
    out_dir: Path,
    meta: dict[str, Any],
) -> None:
    """Item embeddings for the index, user embeddings for every user with history."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = TwoTowerRecommender(model, history_len, device)
    rec.fit(ctx)
    np.save(out_dir / "item_emb.npy", rec.item_emb.float().cpu().numpy())
    has_hist = np.diff(rec.user_start.cpu().numpy()) > 0
    user_emb = np.zeros((ctx.n_users, rec.item_emb.shape[1]), dtype=np.float32)
    users = np.flatnonzero(has_hist)
    for b in range(0, len(users), 8192):
        batch = users[b : b + 8192]
        user_emb[batch] = rec.user_embeddings(batch).float().cpu().numpy()
    np.save(out_dir / "user_emb.npy", user_emb)
    np.save(out_dir / "user_has_history.npy", has_hist)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("exported %s", out_dir)


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    rng = set_seed(cfg["seed"])
    tcfg, ecfg = cfg["two_tower"], cfg["eval"]
    device = resolve_device(tcfg["device"])
    parts, meta = load_split(cfg["paths"]["split_dir"])
    item_map = pd.read_parquet(Path(cfg["paths"]["split_dir"]) / "item_map.parquet")
    feats = build_item_features(cfg["paths"]["processed_dir"], item_map, tcfg["use_genome"])
    val_ctx, test_ctx = build_contexts(parts, meta)
    common = dict(
        retrieval_ks=ecfg["retrieval_ks"],
        ranking_k=ecfg["ranking_k"],
        batch_size=ecfg["batch_size"],
    )
    sample_kwargs = dict(
        common, max_users=tcfg["eval_users_per_epoch"], rng=np.random.default_rng(cfg["seed"])
    )
    tag = run_tag(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    run_info = {"device": device_name(device), "torch": torch.__version__, "two_tower_config": tcfg}

    if tcfg.get("benchmark_steps"):
        _, curve, _ = train(feats, val_ctx, tcfg, device, rng, epochs=1)
        write_result(
            results_dir / f"m2_step_timing_{tag}_{device.type}.json",
            {**run_info, "train_examples_period": "val", "timing": curve[0]},
            cfg,
        )
        return

    t0 = time.perf_counter()
    model, val_curve, best_epoch = train(
        feats,
        val_ctx,
        tcfg,
        device,
        rng,
        tcfg["epochs"],
        eval_ctx=val_ctx,
        eval_kwargs=sample_kwargs,
    )
    val_train_s = time.perf_counter() - t0
    val_res = evaluate(
        TwoTowerRecommender(model, tcfg["history_len"], device), val_ctx, rng=rng, **common
    )
    log.info("val (best epoch %d): %s", best_epoch, val_res)
    art = Path(cfg["paths"].get("artifacts_dir", "artifacts")) / "two_tower" / tag
    base_meta = {
        "epochs": best_epoch,
        "n_items": meta["n_items"],
        "n_users": meta["n_users"],
        **run_info,
    }
    # The val model (trained on train only) generates the ranker's training candidates in M3.
    export(
        model,
        val_ctx,
        tcfg["history_len"],
        device,
        art / "val",
        {"period": "val", "trained_on": "train", **base_meta},
    )

    t0 = time.perf_counter()
    model, test_curve, _ = train(feats, test_ctx, tcfg, device, rng, best_epoch)
    test_train_s = time.perf_counter() - t0
    test_res = evaluate(
        TwoTowerRecommender(model, tcfg["history_len"], device), test_ctx, rng=rng, **common
    )
    log.info("test: %s", test_res)

    export(
        model,
        test_ctx,
        tcfg["history_len"],
        device,
        art / "test",
        {"period": "test", "trained_on": "train+val", **base_meta},
    )

    n_params = sum(p.numel() for p in model.parameters())
    write_result(
        results_dir / f"m2_two_tower_{tag}.json",
        {
            **run_info,
            "n_params": n_params,
            "best_epoch_on_val": best_epoch,
            "val": {"curve": val_curve, "train_s": round(val_train_s, 1), "result": val_res},
            "test": {"curve": test_curve, "train_s": round(test_train_s, 1), "result": test_res},
            "artifacts_dir": str(art),
        },
        cfg,
    )


if __name__ == "__main__":
    main()
