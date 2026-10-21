# M5 summary

## Serving-path quality and in-process cost (all warm test users, 1 thread)

| filter | candidates | NDCG@10 | Recall@candidates | retrieval p50 / p99 ms | ranking p50 / p99 ms | total mean / p99 ms |
|---|---|---|---|---|---|---|
| overfetch | 100 | 0.1434 | 0.2824 | 0.373 / 3.688 | 0.453 / 0.662 | 1.163 / 4.172 |
| overfetch | 50 | 0.1392 | 0.1893 | 0.314 / 3.563 | 0.295 / 0.494 | 0.938 / 3.890 |
| selector | 100 | 0.1428 | 0.2820 | 0.121 / 0.278 | 0.425 / 0.599 | 0.559 / 0.828 |
| selector | 50 | 0.1385 | 0.1889 | 0.118 / 0.282 | 0.277 / 0.439 | 0.412 / 0.662 |

## Load test: before (over-fetch) vs after (FAISS ID selector)

| target rps | before: achieved / client p99 / server retrieval p99 | after: achieved / client p99 / server retrieval p99 |
|---|---|---|
| 50 | 60.0 / 11 ms / 4.728 ms | 60.0 / 8 ms / 1.963 ms |
| 100 | 99.9 / 9 ms / 4.225 ms | 99.9 / 11 ms / 1.492 ms |
| 200 | 199.9 / 14 ms / 2.857 ms | 199.9 / 11 ms / 0.977 ms |
| 400 | 399.7 / 14 ms / 2.53 ms | 399.7 / 16 ms / 0.91 ms |
| 600 | 599.5 / 15 ms / 2.462 ms | 599.4 / 18 ms / 0.798 ms |
| 800 | 799.4 / 22 ms / 2.121 ms | 798.9 / 27 ms / 0.926 ms |
| 1000 | 998.9 / 16 ms / 1.937 ms | 999.2 / 9 ms / 0.493 ms |
| 1200 | 1198.7 / 19 ms / 2.054 ms | 1198.9 / 19 ms / 0.654 ms |
| 1500 | 1497.7 / 20 ms / 1.97 ms | 1496.5 / 24 ms / 0.494 ms |
| 2000 | 1995.4 / 23 ms / 1.985 ms | 1994.4 / 30 ms / 0.567 ms |
| 2500 | 2496.8 / 29 ms / 1.941 ms | 2495.4 / 25 ms / 0.485 ms |
| 3000 | 2997.0 / 37 ms / 2.034 ms | 2986.3 / 48 ms / 0.491 ms |
| 3500 | 3475.0 / 58 ms / 2.229 ms (not sustained) | 3464.2 / 64 ms / 0.494 ms (not sustained) |
| 4000 | 3617.2 / 91 ms / 2.346 ms (not sustained) | 3837.6 / 91 ms / 0.494 ms (not sustained) |

Max sustained throughput (achieved >= 95% of target, client p99 <= 50 ms, failures < 0.1%): before 3000 req/s, after 3000 req/s.

Environment: locust runs in the loadgen container on the compose network (no host port forwarding), pinned to Docker VM vCPUs 5-9; the API is pinned to vCPUs 0-3 and Redis to vCPU 4; all inside one Docker Desktop VM on the same host. API: {'candidates': 100, 'filter_mode': 'overfetch', 'ef_search': 128, 'workers': 4, 'cpus': [0, 1, 2, 3]}. Docker VM (read on the host when summarising): {'cpus': 10, 'mem_gb': 8.2, 'version': '28.5.1'}. Host: {'cpu': 'Apple M4', 'cpu_count': 10, 'ram_gb': 17.2, 'os': 'Darwin 25.5.0', 'python': '3.11.17'}.
