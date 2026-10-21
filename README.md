# recserve

A two-stage movie recommendation service (candidate retrieval, then ranking) on
MovieLens 25M, with a 50 ms p99 latency budget. Work in progress; the full README
(architecture, design decisions, results) comes with milestone M7.

## Status

- [x] M1: data pipeline, temporal split, baselines, evaluation harness
- [x] M2: two-tower retrieval and FAISS HNSW index
- [x] M3: LightGBM LambdaRank ranker
- [x] M4: FastAPI serving with Redis and Prometheus
- [x] M5: load test and optimisation
- [x] M6: simulated A/B test
- [ ] M7: README and results

## Quick start

```bash
conda create -n recserve python=3.11 && conda activate recserve
make install
make data        # downloads ml-25m.zip (262 MB), verifies MD5, writes Parquet (154 MB)
make split       # global temporal split
make baselines   # M1 results -> results/m1_baselines_ml-25m.{json,md}
pip install -e ".[train]"
make two-tower   # M2: train (cuda > mps > cpu), evaluate, export embeddings to artifacts/
make ann         # M2: HNSW vs exact search -> results/m2_ann_ml-25m.{json,md}
make ranker      # M3: LambdaRank on two-tower candidates -> results/m3_ranker_ml-25m.{json,md}
make test        # unit tests + end-to-end smoke test on synthetic data
```

## Serving (M4)

```bash
pip install -e ".[serve]"
make serving-artifacts   # artifacts/serving/ml-25m: HNSW index, user tower, ranker, item tables
make up                  # redis + loader + api (4 workers) + prometheus + grafana
curl "localhost:8000/recommend?user_id=1&k=10"
curl "localhost:8000/recommend?user_id=999999999&k=10&genres=Comedy,Drama"   # cold start
make probe               # sequential latency probe -> results/m4_serving_probe_ml-25m.json
make test-serving        # API tests inside the Linux container
```

Host ports can be changed with `API_PORT`, `REDIS_PORT`, `PROMETHEUS_PORT` and
`GRAFANA_PORT` (for example `GRAFANA_PORT=3001 make up`). Grafana serves the
provisioned `recserve` dashboard at `/d/recserve`.

## Load test and optimisation (M5)

```bash
make up                                   # stack must be running
make serving-quality ARGS="--set serving.filter_mode=overfetch"   # quality + cost per variant
make load-test LABEL=baseline             # locust in the pinned loadgen container
make summarize-m5                         # results/m5_summary_ml-25m.md
```

Locust runs in its own container on the compose network, pinned to different
Docker VM vCPUs than the API (API 0-3, Redis 4, locust 5-9). Two earlier ways of
running it are kept in `results/` because they distorted the numbers: locust on
the macOS host (CPU contention with the VM plus Docker Desktop port forwarding),
and many slow connections (5 req/s per locust user), which at the same total
rate gave several times higher server-side latency than fewer busy connections
(20 req/s per user, the configuration used for the reported numbers).

## Simulated A/B test (M6)

> **This is an offline simulation, not a live experiment.** No user saw these
> recommendations. Held-out test-period ratings stand in for clicks, so there is
> no position bias, no presentation effect and no feedback loop, and MovieLens
> ratings were themselves shaped by the recommender MovieLens ran at the time.

```bash
make ab-test     # in the serving container -> results/m6_ab_ml-25m.json
```

Users with a test-period positive are split 50/50 by a salted SHA-256 of their
userId. Control gets 90-day popularity; treatment gets exactly what the API
returns (`TwoStageRecommender.serve`: two-stage for users with history,
popularity for cold users). A user "clicks" if any of their top 10 is one of
their held-out positives. Power is pre-registered on the val period, analysis
uses the two-proportion z-test and Beta-Binomial model vendored from
[ab-testing-framework](https://github.com/arvindxyogesh/ab-testing-framework)
(`recserve/experiments/abtest_vendored.py`), with an SRM check and 1,000 A/A
re-splits to check calibration.

On macOS the API runs only in Docker: the macOS wheels of faiss and lightgbm (and
torch) each bundle their own OpenMP runtime, and loading two in one process aborts.
Training, the ANN benchmark and serving therefore run in separate processes, and
the serving tests run in CI and in the container.

Every script takes `--config` and `--set key.sub=value` overrides, e.g.
`make split CONFIG=configs/default.yaml` or
`python -m recserve.data.prepare --set data.sample_users=5000`.

## Layout

```
recserve/
  data/         download, indexing and temporal split, synthetic data for tests
  retrieval/    baselines (M1), two-tower model and FAISS index (M2)
  ranking/      feature engineering and LambdaRank (M3)
  serving/      FastAPI service (M4)
  eval/         metrics and the offline evaluation harness
  experiments/  scripts that produce the files in results/
configs/        YAML configs (default = full ML-25M, smoke = synthetic CI run)
results/        JSON and Markdown results, each with config hash, git commit and hardware
```
