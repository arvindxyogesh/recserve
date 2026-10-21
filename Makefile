CONFIG ?= configs/default.yaml
PY ?= python

.PHONY: install data split baselines cold-users two-tower ann smoke test lint format

install:
	$(PY) -m pip install -e ".[dev]"

data:  ## download MovieLens 25M and convert to Parquet
	$(PY) -m recserve.data.download --config $(CONFIG)

split:  ## index users and items, global temporal split
	$(PY) -m recserve.data.prepare --config $(CONFIG)

baselines:  ## M1: random, popularity and co-occurrence baselines
	$(PY) -m recserve.experiments.run_baselines --config $(CONFIG)

ann:  ## M2: HNSW vs exact search benchmark (run on the serving machine)
	$(PY) -m recserve.experiments.ann_benchmark --config $(CONFIG)

two-tower:  ## M2: train, evaluate and export the two-tower model
	$(PY) -m recserve.retrieval.train_two_tower --config $(CONFIG)

smoke:  ## whole pipeline on synthetic data
	$(PY) -m pytest -q -m smoke

test:  ## torch and faiss tests run in separate processes (duplicate libomp on macOS)
	$(PY) -m pytest -q --ignore=tests/faiss
	$(PY) -m pytest -q tests/faiss

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

cold-users:  ## M1: signup-burst analysis behind the cold-user numbers
	$(PY) -m recserve.experiments.cold_user_analysis --config $(CONFIG)
