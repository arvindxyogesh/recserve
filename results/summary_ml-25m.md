# recserve results

All numbers are read from results/*.json by recserve/experiments/report.py.

## Retrieval and ranking (test period, warm users)

| model | users | Recall@100 | NDCG@10 | HitRate@10 |
|---|---|---|---|---|
| popularity | 4727 | 0.1255 | 0.0602 | 0.2513 |
| popularity_90d | 4727 | 0.2216 | 0.0915 | 0.3622 |
| cooccurrence | 4727 | 0.1911 | 0.0919 | 0.3503 |
| two_tower | 4727 | 0.2823 | 0.1143 | 0.4650 |
| two_tower + lambdarank | 4727 | 0.2823 | 0.1436 | 0.5249 |

LambdaRank NDCG@10 lift over retrieval order: +25.7% (95% paired bootstrap CI +21.3% to +30.4%).
LambdaRank NDCG@10 lift over 90-day popularity (full catalogue): +56.9%.

## ANN index (62,423 items, 64-d, 1 thread, k=100)

| index | build s | size MB | p99 ms | overlap with exact |
|---|---|---|---|---|
| flat (exact) | 0.002 | 15.98 | 0.280 | 1.0000 |
| hnsw M=16 efSearch=128 | 3.682 | 24.99 | 0.070 | 0.9794 |

## Serving latency, sequential single client (ms)

| stage | p50 | p99 |
|---|---|---|
| feature_fetch | 0.106 | 0.428 |
| retrieval | 0.185 | 2.947 |
| ranking | 0.429 | 1.179 |
| total | 0.732 | 4.43 |
| client_rtt | 1.732 | 5.617 |

## Load test (locust, 4 API workers on 4 pinned vCPUs)

| variant | max sustained req/s | client p99 at that rate (ms) | server retrieval p99 at that rate (ms) | NDCG@10 (serving path) | in-process mean ms |
|---|---|---|---|---|---|
| over-fetch (before) | 3000 | 37 | 2.034 | 0.1434 | 1.139 |
| FAISS ID selector (after) | 3000 | 48 | 0.491 | 0.1428 | 0.562 |

## Simulated A/B test (offline replay, not a live experiment)

| segment | control | treatment | relative lift (95% CI) | p-value |
|---|---|---|---|---|
| overall | 67.2% (n=5351) | 74.6% (n=5318) | +11.0% (+8.3% to +13.7%) | 4.2e-17 |
| warm | 36.1% (n=2365) | 52.3% (n=2362) | +44.8% (+35.5% to +54.7%) | 4.2e-29 |
| cold | 91.9% (n=2986) | 92.5% (n=2956) | +0.6% (-0.8% to +2.2%) | 0.39 |

SRM p = 0.75; A/A false positive rate 4.4% over 1000 re-splits (alpha 5%).
