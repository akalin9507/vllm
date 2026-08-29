# Disk KV cache benchmark template

This benchmark is intentionally a protocol rather than a synthetic score.
Run the same request trace against two deployments:

1. vLLM with the filesystem secondary tier disabled.
2. vLLM with `TieringOffloadingSpec` and a persistent filesystem cache.

Warm up both deployments with the same requests. Then run the trace twice:
the first pass measures admission and writes; the second pass measures cache
recall. Save `/metrics` and the server log for every run.

Record at minimum:

| Measurement | Required result |
| --- | --- |
| Request count and prompt-token count | Exact values |
| Cache hit/miss count | Per tier |
| Read/write bytes and seconds | From `/metrics` |
| p50/p95/p99 latency | Warm and cold passes |
| Effective prompt throughput | tokens/s |
| Peak and final cache bytes | Filesystem measurement |
| Evicted and rejected blocks | Policy/log measurement |
| Errors and restarts | Count and excerpts |

Example shell outline:

```bash
export BASE=http://localhost:8000
curl -fsS "$BASE/health"
curl -fsS "$BASE/metrics" > metrics.before.txt

# Send the fixed request trace here. Keep JSON bodies under version control,
# but never commit credentials, model files, or the cache volume.

curl -fsS "$BASE/metrics" > metrics.after.txt
du -sb /kv-cache
```

For a useful report, publish the request generator, the exact Docker command,
the two metrics snapshots, and the raw latency samples alongside the summary.
