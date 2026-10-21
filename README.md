# recserve

A two-stage movie recommendation service (candidate retrieval, then ranking) on
MovieLens 25M, with a 50 ms p99 latency budget. Work in progress; the full README
(architecture, design decisions, results) comes with milestone M7.

## Status

- [x] M1: data pipeline, temporal split, baselines, evaluation harness
- [ ] M2: two-tower retrieval and FAISS HNSW index
- [ ] M3: LightGBM LambdaRank ranker
- [ ] M4: FastAPI serving with Redis and Prometheus
- [ ] M5: load test and optimisation
- [ ] M6: simulated A/B test
- [ ] M7: README and results

## Quick start

```bash
conda create -n recserve python=3.11 && conda activate recserve
make install
make data        # downloads ml-25m.zip (262 MB), verifies MD5, writes Parquet (154 MB)
make split       # global temporal split
make baselines   # M1 results -> results/m1_baselines_ml-25m.{json,md}
make test        # unit tests + end-to-end smoke test on synthetic data
```

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
