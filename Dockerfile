# Serving image: FastAPI + FAISS + LightGBM + numpy user tower. No torch.
FROM python:3.11.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1

# libgomp1: OpenMP runtime used by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Dependencies first, so code edits do not reinstall them.
COPY pyproject.toml README.md ./
RUN mkdir recserve && touch recserve/__init__.py && pip install ".[serve]" \
    && pip uninstall -y recserve
COPY recserve ./recserve
COPY configs ./configs
RUN pip install --no-deps .

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && useradd --create-home app
USER app

ENV RECSERVE_ARTIFACTS=/artifacts/serving/ml-25m \
    REDIS_URL=redis://redis:6379/0 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus \
    WEB_CONCURRENCY=4
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
