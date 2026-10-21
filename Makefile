CONFIG ?= configs/default.yaml
PY ?= python

.PHONY: install data split baselines cold-users two-tower ann ranker serving-artifacts up down probe load-test serving-quality test-serving smoke test lint format

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

ranker:  ## M3: LightGBM LambdaRank over two-tower candidates
	$(PY) -m recserve.ranking.train_ranker --config $(CONFIG)

smoke:  ## whole pipeline on synthetic data
	$(PY) -m pytest -q -m smoke

test:  ## torch, faiss and serving tests run in separate processes (duplicate libomp on macOS)
	$(PY) -m pytest -q --ignore=tests/faiss --ignore=tests/serving
	$(PY) -m pytest -q tests/faiss
	@if [ "$$(uname)" = Darwin ]; then echo "tests/serving: run 'make test-serving' (Linux container) on macOS"; \
	else $(PY) -m pytest -q tests/serving; fi

serving-artifacts:  ## M4: package index, user tower, ranker and item tables for the API
	$(PY) -m recserve.serving.build_artifacts --config $(CONFIG)

up:  ## M4: build and start redis, loader, api, prometheus, grafana
	docker compose up -d --build

down:
	docker compose down

probe:  ## M4: sequential latency probe against the running API
	$(PY) -m recserve.experiments.serving_probe --config $(CONFIG)

load-test:  ## M5: locust step load test against the running stack (LABEL=name)
	$(PY) -m recserve.experiments.load_test --config $(CONFIG) --set load_test.label=$(or $(LABEL),baseline)

serving-quality:  ## M5: quality + in-process cost of the serving path, in the container (ARGS=--set ...)
	docker compose build api
	docker compose run --rm --no-deps -v "$(CURDIR)/data/split:/data/split:ro" \
		-v "$(CURDIR)/results:/results" --entrypoint python api \
		-m recserve.experiments.serving_quality --set paths.split_dir=/data/split \
		--set paths.artifacts_dir=/artifacts --set paths.results_dir=/results $(ARGS)

test-serving:  ## serving tests inside the Linux container (needed on macOS); rebuilds the image first
	docker compose build api
	docker compose run --rm --no-deps -v "$(CURDIR)/tests:/app/tests:ro" --entrypoint sh api \
		-c "mkdir -p \$$PROMETHEUS_MULTIPROC_DIR && pip install --user -q pytest httpx fakeredis && python -m pytest -q -p no:cacheprovider tests/serving"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

cold-users:  ## M1: signup-burst analysis behind the cold-user numbers
	$(PY) -m recserve.experiments.cold_user_analysis --config $(CONFIG)
