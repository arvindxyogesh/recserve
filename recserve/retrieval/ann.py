"""FAISS index helpers: exact inner-product search and HNSW."""

from __future__ import annotations

import time

import faiss
import numpy as np


def build_index(
    item_emb: np.ndarray, kind: str, m: int = 32, ef_construction: int = 200
) -> tuple[faiss.Index, float]:
    """Return (index, build seconds). Embeddings are L2-normalised, so IP == cosine."""
    x = np.ascontiguousarray(item_emb, dtype=np.float32)
    d = x.shape[1]
    t0 = time.perf_counter()
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
    else:
        raise ValueError(f"unknown index kind {kind!r}")
    index.add(x)
    return index, time.perf_counter() - t0


def index_bytes(index: faiss.Index) -> int:
    return int(faiss.serialize_index(index).nbytes)


def set_ef_search(index: faiss.Index, ef: int) -> None:
    if isinstance(index, faiss.IndexHNSW):
        index.hnsw.efSearch = ef


def search_filtered(
    index: faiss.Index,
    queries: np.ndarray,
    seen: list[np.ndarray],
    k: int,
    ef: int | None = None,
    max_search: int = 2048,
) -> np.ndarray:
    """Top-k per query after removing already-seen items.

    Retrieves k + |seen| candidates (capped at max_search) so that filtering
    still leaves k items; for HNSW, efSearch is raised to at least that many.
    Rows that end up short are padded with -1.
    """
    out = np.full((len(queries), k), -1, dtype=np.int64)
    for i, (q, s) in enumerate(zip(queries, seen, strict=True)):
        k_search = min(k + len(s), max_search, index.ntotal)
        if ef is not None:
            set_ef_search(index, max(ef, k_search))
        _, ids = index.search(q[None, :], k_search)
        ids = ids[0][~np.isin(ids[0], s)][:k]
        out[i, : len(ids)] = ids
    return out


def latency_ms(index: faiss.Index, queries: np.ndarray, k: int) -> np.ndarray:
    """Per-query wall time (ms) for one query at a time, as a serving request would issue."""
    times = np.empty(len(queries))
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        index.search(q[None, :], k)
        times[i] = (time.perf_counter() - t0) * 1000
    return times
