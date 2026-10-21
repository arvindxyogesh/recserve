"""End-to-end smoke test: synthetic data -> split -> baselines, via the real CLIs."""

import json
import subprocess
import sys

import pytest


@pytest.mark.smoke
def test_pipeline_end_to_end(tmp_path):
    overrides = []
    for key in ("raw_dir", "processed_dir", "split_dir", "results_dir"):
        overrides += ["--set", f"paths.{key}={tmp_path / key}"]
    for module in (
        "recserve.data.synthetic",
        "recserve.data.prepare",
        "recserve.experiments.run_baselines",
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
