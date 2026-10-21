CONFIG ?= configs/default.yaml
PY ?= python

.PHONY: install data split baselines smoke test lint format

install:
	$(PY) -m pip install -e ".[dev]"

data:  ## download MovieLens 25M and convert to Parquet
	$(PY) -m recserve.data.download --config $(CONFIG)

split:  ## index users and items, global temporal split
	$(PY) -m recserve.data.prepare --config $(CONFIG)

baselines:  ## M1: random, popularity and co-occurrence baselines
	$(PY) -m recserve.experiments.run_baselines --config $(CONFIG)

smoke:  ## whole pipeline on synthetic data
	$(PY) -m pytest -q -m smoke

test:
	$(PY) -m pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .
