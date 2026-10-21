"""FastAPI service: GET /recommend?user_id=&k=[&genres=Comedy,Drama]

Stages per request (each has a Prometheus latency histogram):
  feature_fetch  async Redis HGETALL of the user's recent positives, seen items, stats
  retrieval      numpy user tower + FAISS HNSW search + seen-item filtering
  ranking        feature assembly + LightGBM LambdaRank on the candidates
Unknown users, or users with no positive ratings, get the cold-start path:
recent popularity by genre (from ``genres=``) or overall.

Run with several uvicorn workers; with PROMETHEUS_MULTIPROC_DIR set, /metrics
aggregates all workers.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
)

from recserve.serving.recommender import TwoStageRecommender
from recserve.serving.store import KEY, decode

ART_DIR = os.environ.get("RECSERVE_ARTIFACTS", "artifacts/serving/ml-25m")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

BUCKETS = (
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    1.0,
)
STAGE_SECONDS = Histogram(
    "recserve_stage_seconds", "Latency of each request stage", ["stage"], buckets=BUCKETS
)
REQUESTS = Counter("recserve_requests_total", "Recommendation requests", ["source"])

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests inject a recommender and a fake Redis before startup.
    if "rec" not in state:
        state["rec"] = TwoStageRecommender(ART_DIR)
        state["rec"].warmup()
    if "redis" not in state:
        state["redis"] = aioredis.from_url(REDIS_URL)
    yield
    await state["redis"].aclose()


app = FastAPI(title="recserve", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process is up and serving HTTP."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(response: Response) -> dict[str, Any]:
    """Readiness: models loaded and Redis reachable with user state loaded."""
    checks = {"models": "rec" in state, "redis": False, "user_state": False}
    try:
        checks["redis"] = bool(await state["redis"].ping())
        checks["user_state"] = await state["redis"].exists("meta:as_of_ts") == 1
    except Exception:  # noqa: BLE001 - any Redis failure means not ready
        pass
    ready = all(checks.values())
    response.status_code = 200 if ready else 503
    return {"ready": ready, **checks}


@app.get("/metrics")
def metrics() -> Response:
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/recommend")
async def recommend(
    user_id: int,
    k: int = Query(10, ge=1, le=100),
    genres: str | None = Query(None, description="comma-separated, used for cold start"),
) -> dict[str, Any]:
    rec: TwoStageRecommender = state["rec"]
    genre_list = [g for g in (genres or "").split(",") if g]
    unknown = [g for g in genre_list if g not in rec.cold_by_genre]
    if unknown:
        raise HTTPException(422, f"unknown genres {unknown}; valid: {rec.genre_names}")

    t0 = time.perf_counter()
    record = decode(await state["redis"].hgetall(KEY.format(user_id)))
    fetch_s = time.perf_counter() - t0

    if record is not None and len(record.recent):
        result = rec.recommend(record, k)
    else:
        result = rec.cold_start(k, genre_list, None if record is None else record.seen)
    total_s = time.perf_counter() - t0

    STAGE_SECONDS.labels("feature_fetch").observe(fetch_s)
    for stage, seconds in result.timings_s.items():
        STAGE_SECONDS.labels(stage).observe(seconds)
    STAGE_SECONDS.labels("total").observe(total_s)
    REQUESTS.labels(result.source).inc()

    items = [
        {
            "movie_id": int(rec.movie_ids[i]),
            "title": str(rec.titles[i]),
            "score": float(s),
            **({"retrieval_rank": int(r)} if result.retrieval_rank is not None else {}),
        }
        for i, s, r in zip(
            result.items,
            result.scores,
            result.retrieval_rank
            if result.retrieval_rank is not None
            else [None] * len(result.items),
            strict=True,
        )
    ]
    return {
        "user_id": user_id,
        "k": k,
        "source": result.source,
        "items": items,
        "latency_ms": {
            "feature_fetch": round(fetch_s * 1000, 3),
            **{s: round(v * 1000, 3) for s, v in result.timings_s.items()},
            "total": round(total_s * 1000, 3),
        },
    }
