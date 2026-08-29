"""Build and launch a configurable vLLM disk-KV-cache deployment."""

import argparse
import json
import os
import shlex


DEFAULT_MAX_BYTES = 350 * 1024**3


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch vLLM with the filesystem KV secondary tier."
    )
    parser.add_argument("--model", default=os.getenv("KVCATCH_MODEL", "/models"))
    parser.add_argument("--served-model-name")
    parser.add_argument("--tensor-parallel-size", type=positive_int, default=4)
    parser.add_argument("--max-model-len", type=positive_int, default=262144)
    parser.add_argument("--max-num-batched-tokens", type=positive_int, default=2048)
    parser.add_argument("--max-num-seqs", type=positive_int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.91)
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--quantization", default="compressed-tensors")
    parser.add_argument("--attention-backend", default="FlashInfer")
    parser.add_argument("--mtp-tokens", type=non_negative_int, default=4)
    parser.add_argument("--no-speculative", action="store_true")
    parser.add_argument("--cache-root", default="/kv-cache")
    parser.add_argument("--cache-namespace")
    parser.add_argument(
        "--cache-max-bytes", type=non_negative_int, default=DEFAULT_MAX_BYTES
    )
    parser.add_argument("--cache-policy", default="prefix_cost_aware_wtinylfu")
    parser.add_argument("--cache-read-threads", type=positive_int, default=32)
    parser.add_argument("--cache-write-threads", type=positive_int, default=16)
    parser.add_argument(
        "--cache-cpu-bytes", type=non_negative_int, default=12 * 1024**3
    )
    parser.add_argument("--cache-blocks-per-chunk", type=positive_int, default=4)
    parser.add_argument("--cache-recency-half-life", type=float, default=3600.0)
    parser.add_argument("--cache-prefix-weight", type=float, default=1.0)
    parser.add_argument("--cache-prefill-tokens-per-second", type=float, default=1000.0)
    parser.add_argument(
        "--cache-frequency-half-life", type=float, default=3600.0
    )
    parser.add_argument("--cache-window-ratio", type=float, default=0.05)
    parser.add_argument("--cache-probation-ratio", type=float, default=0.20)
    parser.add_argument("--cache-protected-ratio", type=float, default=0.75)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the generated vLLM command"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.cache_policy != "prefix_cost_aware_wtinylfu":
        parser.error("--cache-policy currently supports prefix_cost_aware_wtinylfu")

    ratios = (
        args.cache_window_ratio,
        args.cache_probation_ratio,
        args.cache_protected_ratio,
    )
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        parser.error("cache segment ratios must be non-negative and sum to 1")

    fs_config = {
        "type": "fs",
        "root_dir": args.cache_root,
        "max_bytes": args.cache_max_bytes,
        "cache_policy": args.cache_policy,
        "n_read_threads": args.cache_read_threads,
        "n_write_threads": args.cache_write_threads,
        "recency_half_life_seconds": args.cache_recency_half_life,
        "prefix_weight": args.cache_prefix_weight,
        "prefill_tokens_per_second": args.cache_prefill_tokens_per_second,
        "frequency_sketch_half_life_seconds": args.cache_frequency_half_life,
        "window_ratio": args.cache_window_ratio,
        "probation_ratio": args.cache_probation_ratio,
        "protected_ratio": args.cache_protected_ratio,
    }
    if args.cache_namespace:
        fs_config["cache_namespace"] = args.cache_namespace
    transfer_config = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": args.cache_cpu_bytes,
            "blocks_per_chunk": args.cache_blocks_per_chunk,
            "eviction_policy": "lru",
            "secondary_tiers": [fs_config],
        },
    }
    model_name = args.served_model_name
    if model_name is None:
        model_name = (
            "/models/Qwen3.8-27B-NVFP4"
            if args.model == "/models"
            else args.model
        )
    command = [
        "vllm",
        "serve",
        args.model,
        "--served-model-name",
        model_name,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--quantization",
        args.quantization,
        "--trust-remote-code",
        "--disable-custom-all-reduce",
        "--attention-backend",
        args.attention_backend,
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3",
        "--kv-transfer-config",
        json.dumps(transfer_config, separators=(",", ":")),
    ]
    if not args.no_speculative:
        command.extend(
            [
                "--speculative-config",
                json.dumps(
                    {"method": "mtp", "num_speculative_tokens": args.mtp_tokens},
                    separators=(",", ":"),
                ),
            ]
        )
    command.extend(passthrough)
    if args.dry_run:
        print(shlex.join(command))
        return
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
