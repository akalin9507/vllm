# Configurable vLLM Catch image

Build from the repository root:

```bash
docker build -f docker/Dockerfile.kvcatch \
  -t vllm-catch:v0.28.0-configurable .
```

The entrypoint accepts the most important model, runtime, and filesystem KV
cache settings. Use `--dry-run` to inspect the generated `vllm serve`
command without allocating a GPU:

```bash
docker run --rm vllm-catch:v0.28.0-configurable \
  --model /models/llama \
  --served-model-name llama \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --cache-root /kv-cache \
  --cache-namespace team-a/llama \
  --cache-max-bytes 107374182400 \
  --cache-read-threads 16 \
  --cache-write-threads 8 \
  --dry-run
```

For a real deployment, use the same arguments without `--dry-run` and mount
the model and cache volumes. Additional flags not consumed by the entrypoint
are passed through to `vllm serve`.

## Multiple models on one disk

Run one container per model, sharing a cache root but assigning a namespace and
quota to every model:

```text
/kv-cache/
  team-a/llama/...
  team-a/qwen/...
  team-b/mistral/...
```

The filesystem tier also includes the model identity and run-configuration
digest in its directory name. `cache_namespace` is an operator-visible
partition, not a replacement for that identity check.

`--cache-max-bytes` is a per-namespace limit. If the disk budget is 350 GiB,
configure quotas so their sum is below 350 GiB and leave headroom for SQLite,
temporary files, and unrelated data. The current implementation does not
provide a cross-container global quota coordinator.

Example allocation:

```text
llama:   150 GiB
qwen:    150 GiB
mistral:  40 GiB
headroom: 10 GiB
```

Do not reuse a namespace for incompatible model revisions or KV layouts.
Changing the model revision while keeping the same namespace is safe because
the run-configuration digest changes, but it leaves the old partition until
an operator removes it.
