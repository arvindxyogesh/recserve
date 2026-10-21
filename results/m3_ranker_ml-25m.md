| model | users | ndcg@10 | mrr@10 | hitrate@10 | recall@10 | recall@100 |
|---|---|---|---|---|---|---|
| two_tower (retrieval order) | 4727 | 0.1143 | 0.2216 | 0.4650 | 0.0609 | 0.2823 |
| two_tower + popularity_90d rerank | 4727 | 0.1086 | 0.2053 | 0.4436 | 0.0615 | 0.2823 |
| two_tower + lambdarank | 4727 | 0.1436 | 0.2799 | 0.5249 | 0.0811 | 0.2823 |
| popularity_90d (full catalogue, M1) | 4727 | 0.0915 | 0.1807 | 0.3622 | 0.0485 | 0.2216 |
| cooccurrence (full catalogue, M1) | 4727 | 0.0919 | 0.1844 | 0.3503 | 0.0422 | 0.1911 |
