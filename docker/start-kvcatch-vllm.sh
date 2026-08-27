#!/bin/sh
set -eu

exec vllm serve /models \
    --served-model-name /models/Qwen3.8-27B-NVFP4 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.91 \
    --kv-cache-dtype fp8 \
    --quantization compressed-tensors \
    --trust-remote-code \
    --disable-custom-all-reduce \
    --attention-backend FlashInfer \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
    --reasoning-parser qwen3 \
    --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":12884901888,"blocks_per_chunk":4,"eviction_policy":"lru","secondary_tiers":[{"type":"fs","root_dir":"/kv-cache","max_bytes":375809638400,"cache_policy":"prefix_cost_aware_wtinylfu","recency_half_life_seconds":3600.0,"prefix_weight":1.0,"prefill_tokens_per_second":1000.0,"n_read_threads":32,"n_write_threads":16}]}}'
