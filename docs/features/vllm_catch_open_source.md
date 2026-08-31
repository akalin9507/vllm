# vLLM Catch: Disk KV Cache Tier

This directory contains an experimental filesystem-backed secondary KV cache
for vLLM 0.28. It is intended for long-context and repeated-prefix serving,
where keeping KV blocks on local SSD can reduce repeated prefill work.

## What it provides

- A hard byte limit for persisted `.bin` KV blocks.
- Atomic block writes and corruption-aware reads.
- SQLite WAL metadata with restart reconciliation.
- Asynchronous, read-priority promotion from disk to the primary tier.
- Count-Min Sketch admission and Window/Probation/Protected eviction.
- Prefix-aware, cost-aware retention scoring.

The implementation is experimental. It is not a drop-in replacement for
vLLM's general cache policy and currently targets a local filesystem mounted
into the serving container.

The Docker image enables WSL pinned host memory by default to improve
CPU-to-GPU KV promotion bandwidth. Deployments that encounter WSL/CUDA pinned
memory instability can override `VLLM_WSL2_ENABLE_PIN_MEMORY=0`.

## Reproducible Docker run

Build from the repository root:

```bash
docker build -f docker/Dockerfile.kvcatch -t vllm-catch:v0.28.0-local .
```

The provided `docker/start-kvcatch-vllm.sh` starts the Qwen test deployment.
For another model, copy the Dockerfile and replace the model-specific command
line options. Mount the model read-only and the cache as a persistent volume.

The cache limit is expressed in bytes. For a 350 GiB limit:

```text
max_bytes = 375809638400
```

The Docker entrypoint exposes the main settings as command-line options:

```bash
docker run ... vllm-catch:v0.28.0-local \
  --model /models/llama \
  --served-model-name llama \
  --cache-namespace team-a/llama \
  --cache-max-bytes 107374182400 \
  --cache-read-threads 16 \
  --cache-write-threads 8 \
  --max-model-len 131072
```

Unknown options are passed through to `vllm serve`, so model-specific vLLM
flags can follow the built-in options.

Do not place the SQLite database on a different filesystem from the block
files. The manager reconciles both at startup and removes metadata for files
that no longer exist.

For multiple models, use one shared cache root with a namespace and quota per
model. The model name and run-configuration digest already prevent key
collisions; the explicit namespace makes ownership and quota accounting
visible to operators. `max_bytes` is currently a per-namespace limit. Reserve
headroom and ensure the sum of all namespace limits fits the physical disk.

## Policy model

The default policy is `prefix_cost_aware_wtinylfu`. Its retention score uses
decayed frequency, recency, shared-session count, child count, prefix depth,
estimated prefill savings, observed load cost, block size, and a one-hit
pollution penalty. New admissions are compared with selected victims using
the frequency sketch and score.

The default segment targets are Window 5%, Probation 20%, and Protected 75%.
They are targets for byte pressure, not strict object-count quotas.

## Metrics and benchmark protocol

The vLLM endpoint exposes `/metrics`. Record these metrics before and after a
workload:

- `vllm:kv_offload_tiering_write_bytes`
- `vllm:kv_offload_tiering_write_time`
- `vllm:kv_offload_tiering_read_bytes`
- `vllm:kv_offload_tiering_read_time`
- `vllm:kv_offload_tiering_active_promotion_jobs`
- `vllm:kv_offload_tiering_active_cascade_jobs`

Use the benchmark template in `benchmarks/kvcache/README.md`. Always compare
the disk tier against the same model and request sequence with disk offload
disabled. Report warm-up separately from steady-state requests.

## Known limitations

- The current implementation is optimized for one rank's local filesystem.
- Shared filesystems need a deployment-specific locking and failure policy.
- Segment ratios are configurable but not yet self-tuning.
- Prefill cost begins with a configured throughput estimate and improves as
  observed timings accumulate.
- SQLite writes and startup reconciliation are still areas for scalability
  work at much larger metadata counts.
- The Docker example is model- and GPU-specific; it is not a universal image.

## Release checklist

Before publishing a benchmark result, include the commit, vLLM version,
model revision, GPU type, filesystem type, cache limit, block size, request
trace, hit rates, read/write throughput, p50/p95 latency, and failure count.
