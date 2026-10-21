# Lives in its own directory and pytest process: the macOS wheels of torch and faiss
# each bundle libomp, and loading both in one process aborts.
import numpy as np

from recserve.retrieval.ann import build_index, index_bytes, search_filtered


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_flat_and_hnsw_agree_on_small_data():
    rng = np.random.default_rng(0)
    items = unit(rng.normal(size=(500, 16)))
    queries = unit(rng.normal(size=(20, 16)))
    flat, _ = build_index(items, "flat")
    hnsw, build_s = build_index(items, "hnsw", m=16, ef_construction=200)
    assert build_s >= 0 and index_bytes(hnsw) > index_bytes(flat)
    no_seen = [np.array([], dtype=np.int64)] * len(queries)
    exact = search_filtered(flat, queries, no_seen, 10)
    approx = search_filtered(hnsw, queries, no_seen, 10, ef=200)
    overlap = np.mean([len(np.intersect1d(a, b)) / 10 for a, b in zip(exact, approx, strict=True)])
    assert overlap > 0.95
    np.testing.assert_array_equal(
        exact[0], np.argsort(-(items.astype(np.float64) @ queries[0].astype(np.float64)))[:10]
    )


def test_search_filtered_removes_seen_and_keeps_k():
    rng = np.random.default_rng(1)
    items = unit(rng.normal(size=(100, 8)))
    flat, _ = build_index(items, "flat")
    q = items[:1]
    top = np.argsort(-(items.astype(np.float64) @ q[0].astype(np.float64)))
    seen = [top[:5]]
    out = search_filtered(flat, q, seen, 10)
    assert not np.isin(out[0], seen[0]).any()
    np.testing.assert_array_equal(out[0], top[5:15])
