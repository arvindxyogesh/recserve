"""API tests on small synthetic serving artifacts (no training, no torch).

Skipped on macOS: the macOS wheels of faiss and lightgbm each bundle libomp
and abort when loaded in one process. They run in CI (Linux) and in the
serving container (`make test-serving`).
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

if sys.platform == "darwin":
    pytest.skip("faiss + lightgbm cannot share a process on macOS", allow_module_level=True)

import fakeredis  # noqa: E402
import lightgbm as lgb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from recserve.data.history import PAD  # noqa: E402
from recserve.ranking.features import FEATURES, ITEM_STATS, USER_STATS  # noqa: E402
from recserve.retrieval.ann import build_index  # noqa: E402
from recserve.serving import app as app_module  # noqa: E402
from recserve.serving.build_artifacts import cold_start_lists  # noqa: E402
from recserve.serving.recommender import TwoStageRecommender, right_align  # noqa: E402
from recserve.serving.store import KEY, encode  # noqa: E402

N_ITEMS, DIM, N_GENRES, N_TAGS = 300, 16, 4, 8
GENRES = ["Action", "Comedy", "Drama", "Horror"]


@pytest.fixture(scope="module")
def art(tmp_path_factory):
    rng = np.random.default_rng(0)
    out = tmp_path_factory.mktemp("serving")
    emb = rng.normal(size=(N_ITEMS, DIM)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    index, _ = build_index(emb, "hnsw", 8, 40)
    import faiss

    faiss.write_index(index, str(out / "items.hnsw"))
    genres = (rng.random((N_ITEMS, N_GENRES)) < 0.4).astype(np.float32)
    hidden = 32
    np.savez(
        out / "user_tower.npz",
        item_id=np.vstack([rng.normal(size=(N_ITEMS, DIM)), np.zeros((1, DIM))]).astype(np.float32),
        genres=np.vstack([genres, np.zeros((1, N_GENRES))]).astype(np.float32),
        w0=rng.normal(size=(hidden, DIM + N_GENRES + 1)).astype(np.float32),
        b0=np.zeros(hidden, np.float32),
        w1=rng.normal(size=(DIM, hidden)).astype(np.float32),
        b1=np.zeros(DIM, np.float32),
    )
    x = rng.random((2000, len(FEATURES)))
    y = (rng.random(2000) < 0.1).astype(int)
    booster = lgb.train(
        {"objective": "lambdarank", "verbose": -1, "min_data_in_leaf": 5},
        lgb.Dataset(x, y, group=[100] * 20),
        5,
    )
    booster.save_model(out / "ranker.txt")
    stats = rng.random((N_ITEMS, len(ITEM_STATS))).astype(np.float32) * 100
    np.save(out / "item_stats.npy", stats)
    np.save(out / "item_genres.npy", genres)
    np.save(out / "item_year.npy", rng.integers(1950, 2019, N_ITEMS).astype(np.float32))
    np.save(out / "item_has_genome.npy", np.ones(N_ITEMS, np.float32))
    g = rng.random((N_ITEMS, N_TAGS)).astype(np.float32)
    np.save(out / "genome_unit.npy", g / np.linalg.norm(g, axis=1, keepdims=True))
    pop = stats[:, ITEM_STATS.index("i_pop_90d")]
    (out / "cold_start.json").write_text(json.dumps(cold_start_lists(pop, genres, GENRES, 50)))
    pd.DataFrame(
        {
            "item_idx": np.arange(N_ITEMS),
            "movieId": np.arange(N_ITEMS) + 1000,
            "title": [f"m{i}" for i in range(N_ITEMS)],
            "genres": "x",
        }
    ).to_parquet(out / "items.parquet")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "serving_config": {"candidates": 50, "ef_search": 64, "max_search": 2048},
                "two_tower_history_len": 10,
                "ranker_history_len": 10,
                "genre_names": GENRES,
            }
        )
    )
    return out


@pytest.fixture
def client(art):
    fake = fakeredis.aioredis.FakeRedis()
    app_module.state.clear()
    app_module.state["rec"] = TwoStageRecommender(art)
    app_module.state["rec"].warmup()
    app_module.state["redis"] = fake

    async def seed():
        stats = np.ones(len(USER_STATS))
        await fake.hset(KEY.format(7), mapping=encode(np.arange(5), np.arange(20), stats))
        # user 8 rated a few items but has no positives
        await fake.hset(KEY.format(8), mapping=encode(np.array([], int), np.arange(3), stats))
        await fake.set("meta:as_of_ts", 1)

    with TestClient(app_module.app) as c:
        c.portal.call(seed)
        yield c
    app_module.state.clear()


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["ready"]


def test_two_stage_excludes_seen_and_reports_stages(client):
    r = client.get("/recommend", params={"user_id": 7, "k": 10}).json()
    assert r["source"] == "two_stage"
    assert len(r["items"]) == 10
    ids = [it["movie_id"] - 1000 for it in r["items"]]
    assert not set(ids) & set(range(20))  # seen items are filtered
    scores = [it["score"] for it in r["items"]]
    assert scores == sorted(scores, reverse=True)
    assert set(r["latency_ms"]) == {"feature_fetch", "retrieval", "ranking", "total"}


def test_cold_start_unknown_user(client):
    r = client.get("/recommend", params={"user_id": 999, "k": 5}).json()
    assert r["source"] == "cold_start_global" and len(r["items"]) == 5
    r = client.get("/recommend", params={"user_id": 999, "k": 5, "genres": "Drama"}).json()
    assert r["source"] == "cold_start_genre"
    assert client.get("/recommend", params={"user_id": 1, "genres": "Nope"}).status_code == 422


def test_cold_start_user_without_positives_skips_seen(client):
    r = client.get("/recommend", params={"user_id": 8, "k": 20}).json()
    assert r["source"] == "cold_start_global"
    assert not {it["movie_id"] - 1000 for it in r["items"]} & {0, 1, 2}


def test_k_validation(client):
    assert client.get("/recommend", params={"user_id": 7, "k": 0}).status_code == 422
    assert client.get("/recommend", params={"user_id": 7, "k": 101}).status_code == 422


def test_metrics_exposes_stage_histograms(client):
    client.get("/recommend", params={"user_id": 7})
    body = client.get("/metrics").text
    for stage in ("feature_fetch", "retrieval", "ranking", "total"):
        assert f'recserve_stage_seconds_bucket{{le="0.0002",stage="{stage}"}}' in body
    assert "recserve_requests_total" in body


def test_right_align():
    assert right_align(np.array([1, 2, 3]), 5).tolist() == [[PAD, PAD, 1, 2, 3]]
    assert right_align(np.array([1, 2, 3]), 2).tolist() == [[2, 3]]
    assert right_align(np.array([], dtype=np.int64), 2).tolist() == [[PAD, PAD]]
