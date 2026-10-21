"""End-to-end smoke test: synthetic data -> split -> baselines, via the real CLIs."""

import json
import subprocess
import sys

import pytest


@pytest.mark.smoke
def test_pipeline_end_to_end(tmp_path):
    overrides = []
    for key in ("raw_dir", "processed_dir", "split_dir", "results_dir", "artifacts_dir"):
        overrides += ["--set", f"paths.{key}={tmp_path / key}"]
    for module in (
        "recserve.data.synthetic",
        "recserve.data.prepare",
        "recserve.experiments.run_baselines",
        "recserve.retrieval.train_two_tower",
        "recserve.experiments.ann_benchmark",
    ):
        subprocess.run(
            [sys.executable, "-m", module, "--config", "configs/smoke.yaml", *overrides], check=True
        )

    result = json.loads((tmp_path / "results_dir" / "m1_baselines_synthetic.json").read_text())
    warm = {
        (r["period"], r["model"]): r["metrics"]
        for r in result["results"]
        if r["users_kind"] == "warm"
    }
    assert {"random", "popularity", "cooccurrence"} <= {m for _, m in warm}
    for period in ("val", "test"):
        # sanity: real baselines beat random on structured synthetic data
        assert warm[(period, "popularity")]["recall@50"] > warm[(period, "random")]["recall@50"]
        assert warm[(period, "cooccurrence")]["ndcg@10"] > warm[(period, "random")]["ndcg@10"]
        for value in warm[(period, "cooccurrence")].values():
            assert 0.0 <= value <= 1.0

    tt = json.loads((tmp_path / "results_dir" / "m2_two_tower_synthetic.json").read_text())
    assert 1 <= tt["best_epoch_on_val"] <= 2
    assert tt["test"]["result"]["metrics"]["recall@50"] > warm[("test", "random")]["recall@50"]
    for period in ("val", "test"):
        art = tmp_path / "artifacts_dir" / "two_tower" / "synthetic" / period
        for name in ("item_emb.npy", "user_emb.npy", "user_has_history.npy", "model.pt"):
            assert (art / name).exists()

    ann = json.loads((tmp_path / "results_dir" / "m2_ann_synthetic.json").read_text())
    flat = ann["results"][0]
    assert flat["index"] == "flat (exact)"
    # exact FAISS search with seen-item filtering must reproduce the harness number
    assert flat["recall@50_test"] == pytest.approx(tt["test"]["result"]["metrics"]["recall@50"])
