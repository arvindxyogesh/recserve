#!/bin/sh
set -e
if [ "$1" = "serve" ]; then
  # Prometheus multiprocess mode: one shared directory, emptied on start.
  rm -rf "$PROMETHEUS_MULTIPROC_DIR" && mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
  exec uvicorn recserve.serving.app:app --host 0.0.0.0 --port 8000 \
    --workers "${WEB_CONCURRENCY:-4}" --no-access-log
fi
exec "$@"
