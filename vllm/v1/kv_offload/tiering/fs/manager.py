# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
import threading
from collections.abc import Iterable
from time import monotonic, time
from typing import TYPE_CHECKING, Any, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingCounterMetadata,
    OffloadingEvent,
    OffloadingGaugeMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_load_block,
    batch_store_block,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.policy import (
    POLICY_VERSION,
    CacheEntry,
    CacheMetadataStore,
    PrefixCostAwareWTinyLFU,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        if _HAS_BATCH_LOOKUP_C:
            # C extension: GIL released for the entire faccessat() batch.
            return batch_lookup_C(paths)
        return (os.path.exists(p) for p in paths)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        KV cache sharing between multiple vLLM instances using the same
        ``root_dir`` (e.g., via a shared PVC) works by default: ``NONE_HASH``
        (the chain-hash seed for block content hashes) is derived from a fixed
        default seed, so identical token content produces identical block
        filenames across instances. Setting the ``PYTHONHASHSEED`` environment
        variable to the same value on all instances overrides the default seed,
        and is required to share a cache when using a non-cryptographic
        prefix-caching hash algorithm, which seeds ``NONE_HASH`` randomly.
    """

    medium: ClassVar[Medium] = Medium.STORAGE

    CACHE_BYTES = "vllm:kv_offload_fs_cache_bytes"
    CACHE_ENTRIES = "vllm:kv_offload_fs_cache_entries"
    ADMISSION_ATTEMPTS = "vllm:kv_offload_fs_admission_attempts"
    ADMITTED_BLOCKS = "vllm:kv_offload_fs_admitted_blocks"
    REJECTED_BLOCKS = "vllm:kv_offload_fs_rejected_blocks"
    PARTIAL_ADMISSION_BATCHES = "vllm:kv_offload_fs_partial_admission_batches"
    SKIPPED_BATCHES = "vllm:kv_offload_fs_skipped_batches"
    EVICTIONS = "vllm:kv_offload_fs_evictions"
    EVICTED_BYTES = "vllm:kv_offload_fs_evicted_bytes"
    ADMISSION_REJECTIONS = "vllm:kv_offload_fs_admission_rejections"

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        """Return capacity, admission, and eviction metric definitions."""
        del extra_config
        counter_docs = {
            cls.ADMISSION_ATTEMPTS: "Filesystem cache admission attempts.",
            cls.ADMITTED_BLOCKS: "Blocks admitted to the filesystem cache.",
            cls.REJECTED_BLOCKS: "Blocks rejected by filesystem cache admission.",
            cls.PARTIAL_ADMISSION_BATCHES: (
                "Store batches admitted only partially to the filesystem cache."
            ),
            cls.SKIPPED_BATCHES: (
                "Store batches skipped because no block was admitted."
            ),
            cls.EVICTIONS: "Blocks evicted from the filesystem cache.",
            cls.EVICTED_BYTES: "Bytes evicted from the filesystem cache.",
        }
        definitions: dict[str, OffloadingMetricMetadata] = {
            name: OffloadingCounterMetadata(documentation=documentation)
            for name, documentation in counter_docs.items()
        }
        definitions[cls.CACHE_BYTES] = OffloadingGaugeMetadata(
            documentation="Current filesystem KV cache block-data bytes."
        )
        definitions[cls.CACHE_ENTRIES] = OffloadingGaugeMetadata(
            documentation="Current filesystem KV cache block entry count."
        )
        definitions[cls.ADMISSION_REJECTIONS] = OffloadingCounterMetadata(
            documentation="Filesystem cache admission rejections by reason.",
            labelnames=("reason",),
        )
        return definitions

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
        max_bytes: int | None = None,
        cache_policy: str = "prefix_cost_aware_wtinylfu",
        recency_half_life_seconds: float = 3600.0,
        prefix_weight: float = 1.0,
        prefill_tokens_per_second: float = 1000.0,
        frequency_sketch_half_life_seconds: float = 3600.0,
        window_ratio: float = 0.05,
        probation_ratio: float = 0.20,
        protected_ratio: float = 0.75,
        cache_namespace: str | None = None,
    ):
        """
        Args:
            offloading_spec: Contains normalized offloading configuration and
                blocks_per_chunk.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
            max_bytes: Maximum bytes of block data stored by this rank. ``None``
                keeps the historical unlimited behavior. The limit excludes
                ``config.json`` and temporary files.
            cache_policy: Admission and eviction policy for disk entries.
            recency_half_life_seconds: Half-life used by the policy's recency
                and frequency decay.
            prefix_weight: Weight for shallow/shared prefix entries.
            prefill_tokens_per_second: Initial estimate for prefill cost.
            cache_namespace: Optional relative partition below ``root_dir``.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        if max_bytes is not None and (
            isinstance(max_bytes, bool) or not isinstance(max_bytes, int)
        ):
            raise TypeError("max_bytes must be a non-negative integer or None")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer or None")
        if cache_policy != "prefix_cost_aware_wtinylfu":
            raise ValueError("cache_policy must be 'prefix_cost_aware_wtinylfu'")
        self.max_bytes = max_bytes
        self.cache_policy = cache_policy
        self.locality = Locality(locality) if locality is not None else None

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Keys of in-flight load (promotion) jobs, so a failed load can mark
        # its own cached lookup verdicts False (see get_finished_jobs).
        self._load_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Per load job: how many blocks loaded before a failure (partial keep).
        # Written by the pool worker inside the load task before it raises (so
        # before task_done publishes the job); read on the scheduler thread in
        # get_finished_jobs only for job ids the finished queue returned. Under
        # the GIL that read cannot observe the finished job without the prior
        # write, so no extra lock is needed (get_finished is itself lock-free).
        self._load_progress: dict[JobId, int] = {}
        self._store_evictions: dict[JobId, list[OffloadKey]] = {}
        self._load_paths: dict[JobId, list[str]] = {}
        self._last_policy_age = time()

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
            cache_namespace=cache_namespace,
        )
        self._storage_dir = f"{self.file_mapper.base_path}_r{self.file_mapper.rank}"
        self._capacity_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._entries: dict[str, int] = {}
        self._path_to_key: dict[str, OffloadKey] = {}
        self._protected_load_paths: set[str] = set()
        self._request_group_tails: dict[tuple[str, int], str] = {}
        self._stats = OffloadingConnectorStats()
        self._policy = PrefixCostAwareWTinyLFU(
            recency_half_life_seconds=recency_half_life_seconds,
            prefix_weight=prefix_weight,
            prefill_tokens_per_second=prefill_tokens_per_second,
            frequency_sketch_half_life_seconds=(frequency_sketch_half_life_seconds),
            window_ratio=window_ratio,
            probation_ratio=probation_ratio,
            protected_ratio=protected_ratio,
        )
        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        self._metadata = CacheMetadataStore(f"{self._storage_dir}.metadata.sqlite3")
        self._policy.load(self._metadata.load())
        self._metadata.set_policy_version(POLICY_VERSION)

        with self._capacity_lock:
            self._refresh_entries_locked()
            self._flush_metadata_locked()

        # Prefer O_DIRECT to bypass the page cache, but fall back to buffered
        # I/O on filesystems that reject it (e.g. overlayfs, some NFS mounts)
        # rather than failing every block.
        self._use_o_direct = probe_o_direct(os.path.dirname(config_path))
        if not self._use_o_direct:
            logger.warning(
                "O_DIRECT is not supported at '%s'; falling back to buffered "
                "I/O for the '%s' KV offload tier.",
                root_dir,
                tier_type,
            )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        with self._capacity_lock:
            self._policy.record_access(key.hex(), session_id=req_context.req_id)
            if result:
                self._request_group_tails[
                    (req_context.req_id, get_offload_group_idx(key))
                ] = key.hex()
        if result:
            self._touch_paths(
                [self.file_mapper.get_file_name(key)], record_policy=False
            )
            return LookupResult.HIT
        return LookupResult.MISS

    def _parse_key(self, path: str) -> OffloadKey | None:
        """Recover an offload key from a path in this tier's layout."""
        try:
            group_dir = os.path.basename(os.path.dirname(path))
            hash_hex = os.path.basename(path)[:-4]
            group_idx = int(group_dir.rsplit("_g", 1)[1])
            return make_offload_key(bytes.fromhex(hash_hex), group_idx)
        except (IndexError, ValueError):
            return None

    def _refresh_entries_locked(self) -> None:
        """Reconcile files with policy metadata after restart or eviction."""
        files: dict[str, tuple[OffloadKey, int]] = {}
        if os.path.isdir(self._storage_dir):
            for dirpath, _, filenames in os.walk(self._storage_dir):
                for filename in filenames:
                    if not filename.endswith(".bin"):
                        continue
                    path = os.path.join(dirpath, filename)
                    try:
                        stat = os.stat(path)
                    except OSError:
                        continue
                    key = self._parse_key(path)
                    if key is not None:
                        files[path] = (key, stat.st_size)

        current_keys = {key.hex() for key, _ in files.values()}
        stale_keys = [key for key in self._policy.entries if key not in current_keys]
        for cache_key in stale_keys:
            self._policy.on_remove(cache_key)
        if stale_keys:
            self._metadata.delete(stale_keys)

        self._entries = {}
        self._path_to_key = {}
        for path, (key, size) in files.items():
            cache_key = key.hex()
            entry = self._policy.entries.get(cache_key)
            if entry is None:
                entry = self._policy.make_entry(
                    cache_key=cache_key,
                    path=path,
                    size_bytes=size,
                    token_count=self._token_count(key),
                )
                entry.created_at = entry.last_access_at = os.path.getmtime(path)
                self._policy.on_store(entry, now=entry.last_access_at)
            else:
                entry.path = path
                entry.size_bytes = size
                entry.state = (
                    "READING" if path in self._protected_load_paths else "READY"
                )
            self._entries[path] = size
            self._path_to_key[path] = key
        self._reconcile_parent_links_locked()

    def _reconcile_parent_links_locked(self) -> None:
        """Rebuild parent links and child counts from authoritative entries."""
        entries = self._policy.entries
        states: dict[str, int] = {}

        def normalize(cache_key: str) -> None:
            state = states.get(cache_key, 0)
            entry = entries[cache_key]
            if state == 2:
                return
            if state == 1:
                entry.parent_key = None
                entry.depth = 0
                entry.token_start = 0
                entry.token_end = entry.token_count
                states[cache_key] = 2
                return

            states[cache_key] = 1
            parent_key = entry.parent_key
            if parent_key is None or parent_key not in entries:
                entry.parent_key = None
                entry.depth = 0
                entry.token_start = 0
                entry.token_end = entry.token_count
            else:
                normalize(parent_key)
                parent = entries.get(parent_key)
                if parent is None:
                    entry.parent_key = None
                    entry.depth = 0
                    entry.token_start = 0
                    entry.token_end = entry.token_count
                else:
                    entry.depth = parent.depth + 1
                    entry.token_start = parent.token_end
                    entry.token_end = entry.token_start + entry.token_count
            states[cache_key] = 2

        for cache_key in entries:
            normalize(cache_key)
        for entry in entries.values():
            entry.child_count = 0
        for entry in entries.values():
            if entry.parent_key is not None:
                parent = entries.get(entry.parent_key)
                if parent is not None:
                    parent.child_count += 1

    def _flush_metadata_locked(self) -> None:
        """Batch-persist policy metadata; callers hold the capacity lock."""
        self._metadata.save(self._policy.entries.values())

    def _touch_paths(
        self,
        paths: Iterable[str],
        session_id: str | None = None,
        record_policy: bool = True,
    ) -> None:
        """Record access without using file timestamps as the policy."""
        with self._capacity_lock:
            for path in paths:
                try:
                    size = os.stat(path).st_size
                except OSError:
                    continue
                self._entries[path] = size
                key = self._path_to_key.get(path)
                if key is not None and record_policy:
                    self._policy.record_access(key.hex(), session_id=session_id)

    def _remove_entry_locked(self, path: str) -> OffloadKey | None:
        """Remove one block file and its bookkeeping, if it is removable."""
        try:
            os.remove(path)
        except FileNotFoundError:
            self._entries.pop(path, None)
            key = self._path_to_key.pop(path, None)
            if key is not None:
                self._policy.on_remove(key.hex())
                self._metadata.delete((key.hex(),))
                self._drop_request_tails_locked(key.hex())
            return key
        except OSError:
            return None
        self._entries.pop(path, None)
        key = self._path_to_key.pop(path, None)
        if key is not None:
            self._policy.on_remove(key.hex())
            self._metadata.delete((key.hex(),))
            self._drop_request_tails_locked(key.hex())
        return key

    def _drop_request_tails_locked(self, cache_key: str) -> None:
        stale_tails = [
            tail
            for tail, value in self._request_group_tails.items()
            if value == cache_key
        ]
        for tail in stale_tails:
            del self._request_group_tails[tail]

    def _increase_counter(
        self,
        name: str,
        value: int | float = 1,
        labelvalues: tuple[str, ...] = (),
    ) -> None:
        self._stats.increase_counter(name, value, labelvalues)

    def _record_admission_rejection(self, reason: str) -> None:
        self._increase_counter(
            self.ADMISSION_REJECTIONS,
            labelvalues=(reason,),
        )

    def _token_count(self, key: OffloadKey) -> int:
        """Return the token span represented by one persisted chunk."""
        group_idx = get_offload_group_idx(key)
        tokens_per_block = getattr(self._offloading_spec, "tokens_per_block", ())
        if group_idx < len(tokens_per_block):
            return (
                int(tokens_per_block[group_idx])
                * self._offloading_spec.blocks_per_chunk
            )
        return (
            int(self._offloading_spec.tokens_per_hash)
            * self._offloading_spec.blocks_per_chunk
        )

    def _make_room_locked(
        self, paths: list[str], session_id: str | None = None
    ) -> tuple[list[str], list[OffloadKey]]:
        """Admit as many blocks as fit while preserving prefix chains."""
        unique_paths = list(dict.fromkeys(paths))
        existing_paths = [path for path in unique_paths if os.path.exists(path)]
        missing_paths = [path for path in unique_paths if not os.path.exists(path)]
        self._increase_counter(self.ADMISSION_ATTEMPTS, len(missing_paths))
        if self.max_bytes is None:
            self._increase_counter(self.ADMITTED_BLOCKS, len(missing_paths))
            return unique_paths, []

        self._refresh_entries_locked()
        current_bytes = sum(self._entries.values())
        protected = self._protected_load_paths | set(unique_paths)
        group_tails = {
            group: tail
            for (request_id, group), tail in self._request_group_tails.items()
            if request_id == session_id
        }
        admitted_missing: list[str] = []
        blocked_groups: set[int] = set()

        def select_victims(required_blocks: int) -> list[CacheEntry]:
            bytes_to_free = max(
                current_bytes + required_blocks * self._block_size - self.max_bytes,
                0,
            )
            if bytes_to_free == 0:
                return []
            candidates = [
                self._policy.entries[key.hex()]
                for path, key in self._path_to_key.items()
                if path not in protected
                and key.hex() in self._policy.entries
                and self._policy.entries[key.hex()].state == "READY"
            ]
            return self._policy.select_victims(candidates, bytes_to_free)

        for path in missing_paths:
            key = self._parse_key(path)
            if key is None:
                self._record_admission_rejection("invalid_key")
                continue
            group_idx = get_offload_group_idx(key)
            if group_idx in blocked_groups:
                self._record_admission_rejection("prefix_chain")
                continue
            parent_key = group_tails.get(group_idx)
            candidate = self._policy.make_entry(
                cache_key=key.hex(),
                path=path,
                size_bytes=self._block_size,
                token_count=self._token_count(key),
                parent_key=parent_key,
                session_id=session_id,
            )
            victims = select_victims(len(admitted_missing) + 1)
            required_bytes = max(
                current_bytes
                + (len(admitted_missing) + 1) * self._block_size
                - self.max_bytes,
                0,
            )
            if (
                required_bytes > 0
                and sum(victim.size_bytes for victim in victims) < required_bytes
            ):
                blocked_groups.add(group_idx)
                self._record_admission_rejection("capacity")
                continue
            if required_bytes > 0 and not self._policy.should_admit(candidate, victims):
                blocked_groups.add(group_idx)
                self._record_admission_rejection("policy")
                continue
            admitted_missing.append(path)
            group_tails[group_idx] = key.hex()

        accepted_paths = existing_paths + admitted_missing
        if admitted_missing:
            self._increase_counter(self.ADMITTED_BLOCKS, len(admitted_missing))
        rejected_count = len(missing_paths) - len(admitted_missing)
        if rejected_count:
            self._increase_counter(self.REJECTED_BLOCKS, rejected_count)
        if admitted_missing and rejected_count:
            self._increase_counter(self.PARTIAL_ADMISSION_BATCHES)
        elif missing_paths and not admitted_missing:
            self._increase_counter(self.SKIPPED_BATCHES)

        final_victims = select_victims(len(admitted_missing))
        required_bytes = max(
            current_bytes + len(admitted_missing) * self._block_size - self.max_bytes,
            0,
        )
        if (
            required_bytes > 0
            and sum(victim.size_bytes for victim in final_victims) < required_bytes
        ):
            accepted_paths = existing_paths
            admitted_missing.clear()
            final_victims = []

        evicted: list[OffloadKey] = []
        evicted_bytes = 0
        for entry in final_victims:
            key = self._remove_entry_locked(entry.path)
            if key is not None:
                evicted.append(key)
                evicted_bytes += entry.size_bytes
        if evicted:
            self._increase_counter(self.EVICTIONS, len(evicted))
            self._increase_counter(self.EVICTED_BYTES, evicted_bytes)
        return accepted_paths, evicted

    def _store_batch(
        self,
        job_id: JobId,
        paths: list[str],
        offsets: list[int],
        session_id: str | None,
    ) -> None:
        """Admit and write one batch while serializing capacity decisions."""
        with self._capacity_lock:
            admitted_paths, evicted = self._make_room_locked(paths, session_id)
            if self.events is not None:
                self._store_evictions[job_id] = evicted
            admitted_set = set(admitted_paths)
            seen_paths: set[str] = set()
            batch_paths: list[str] = []
            batch_offsets: list[int] = []
            for path, offset in zip(paths, offsets):
                if path in admitted_set and path not in seen_paths:
                    batch_paths.append(path)
                    batch_offsets.append(offset)
                    seen_paths.add(path)
            if not batch_paths:
                return
            started_at = monotonic()
            try:
                batch_store_block(
                    batch_paths,
                    self._primary_kv_view,
                    batch_offsets,
                    self._block_size,
                    self._use_o_direct,
                )
                store_ms = max((monotonic() - started_at) * 1000.0, 0.001)
                per_entry_store_ms = store_ms / max(len(batch_paths), 1)
                for path in batch_paths:
                    key = self._parse_key(path)
                    if key is None or not os.path.exists(path):
                        continue
                    group_idx = get_offload_group_idx(key)
                    tail_key = None
                    if session_id is not None:
                        tail_key = self._request_group_tails.get(
                            (session_id, group_idx)
                        )
                    entry = self._policy.make_entry(
                        cache_key=key.hex(),
                        path=path,
                        size_bytes=os.path.getsize(path),
                        token_count=self._token_count(key),
                        parent_key=tail_key,
                        session_id=session_id,
                    )
                    previous = self._policy.entries.get(key.hex())
                    previous_store_ms = (
                        0.0 if previous is None else previous.observed_store_ms_ema
                    )
                    entry.observed_store_ms_ema = (
                        per_entry_store_ms
                        if previous_store_ms <= 0
                        else 0.8 * previous_store_ms + 0.2 * per_entry_store_ms
                    )
                    self._policy.on_store(entry)
                    if session_id is not None:
                        self._request_group_tails[(session_id, group_idx)] = key.hex()
                    self._entries[path] = entry.size_bytes
                    self._path_to_key[path] = key
            finally:
                self._reconcile_parent_links_locked()
                self._flush_metadata_locked()

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        keys = list(job_metadata.keys)
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = keys
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        offsets = [int(bid) * self._block_size for bid in job_metadata.block_ids]
        task = functools.partial(
            self._store_batch,
            job_metadata.job_id,
            paths,
            offsets,
            job_metadata.req_context.req_id,
        )
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        job_id = job_metadata.job_id
        # Track this load's keys so a failed promotion can mark only its failed
        # keys as a miss (see get_finished_jobs).
        keys = list(job_metadata.keys)
        self._load_job_keys[job_id] = keys
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        self._load_paths[job_id] = paths
        with self._capacity_lock:
            self._protected_load_paths.update(paths)
            for path in paths:
                key = self._path_to_key.get(path)
                if key is not None and key.hex() in self._policy.entries:
                    self._policy.entries[key.hex()].state = "READING"
        offsets = [int(bid) * self._block_size for bid in job_metadata.block_ids]

        def load_task() -> None:
            try:
                batch_load_block(
                    paths,
                    self._primary_kv_view,
                    offsets,
                    self._block_size,
                    self._use_o_direct,
                )
            except OSError as exc:
                # Runs on the pool worker thread. Record how many blocks loaded
                # before the failure so get_finished_jobs can keep them; this
                # write precedes task_done, so the scheduler reads it safely
                # under the GIL once the finished queue hands back this job.
                num_succeeded = getattr(exc, "num_succeeded", 0)
                self._load_progress[job_id] = num_succeeded
                # Surfaces errno (e.g. EMFILE "Too many open files") for both
                # the C and Python load paths.
                logger.debug(
                    "Load of %d blocks for job %s failed at block %d: %s",
                    len(paths),
                    job_id,
                    num_succeeded,
                    exc,
                )
                raise

        self._pool.enqueue_load(job_id, 1, [load_task])

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Collect finished jobs; a failed promotion marks only its failed keys
        as a miss here (scheduler thread)."""
        results = []
        for job_id, success, transfer_time in self._pool.get_finished():
            with self._capacity_lock:
                load_paths = self._load_paths.pop(job_id, [])
                for path in load_paths:
                    self._protected_load_paths.discard(path)
                    key = self._path_to_key.get(path)
                    if key is None:
                        continue
                    entry = self._policy.entries.get(key.hex())
                    if entry is None:
                        continue
                    if os.path.exists(path):
                        entry.state = "READY"
                        if transfer_time is not None:
                            sample_ms = transfer_time * 1000 / max(len(load_paths), 1)
                            entry.observed_load_ms_ema = (
                                0.2 * sample_ms + 0.8 * entry.observed_load_ms_ema
                            )
                    else:
                        self._policy.on_remove(key.hex())
                self._reconcile_parent_links_locked()
                evicted_keys = self._store_evictions.pop(job_id, [])
                self._flush_metadata_locked()
            if self.events is not None:
                if evicted_keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=evicted_keys,
                            medium=self.medium,
                            removed=True,
                            locality=self.locality,
                        )
                    )
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    keys = [
                        key
                        for key in keys
                        if os.path.exists(self.file_mapper.get_file_name(key))
                    ]
                if success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            load_keys = self._load_job_keys.pop(job_id, None)
            num_succeeded = self._load_progress.pop(job_id, 0)
            if load_keys is not None and not success:
                # A batched load stops at the first bad block and reports how
                # many loaded before it. Those earlier blocks are kept in the
                # primary tier (reported via successful_keys); only this block
                # and the ones after it are marked a miss and recomputed.
                successful = load_keys[:num_succeeded]
                failed = load_keys[num_succeeded:]
                self._lookup_manager.mark_miss(failed)
                results.append(
                    JobResult(
                        job_id=job_id,
                        success=False,
                        successful_keys=tuple(successful) if successful else None,
                        transfer_time=transfer_time,
                    )
                )
                continue
            results.append(
                JobResult(
                    job_id=job_id,
                    success=success,
                    transfer_time=transfer_time,
                )
            )
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @property
    def disk_usage_bytes(self) -> int:
        """Return the current block-data usage for this rank's cache."""
        with self._capacity_lock:
            self._refresh_entries_locked()
            return sum(self._entries.values())

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        """Return filesystem capacity and policy observations since last call."""
        with self._capacity_lock:
            self._refresh_entries_locked()
            self._stats.set_gauge(self.CACHE_BYTES, sum(self._entries.values()))
            self._stats.set_gauge(self.CACHE_ENTRIES, len(self._entries))
            if self._stats.is_empty():
                return None
            stats = self._stats
            self._stats = OffloadingConnectorStats()
            return stats

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        key_list = list(keys)
        with self._capacity_lock:
            for key in key_list:
                self._policy.record_access(key.hex(), session_id=req_context.req_id)
                self._request_group_tails[
                    (req_context.req_id, get_offload_group_idx(key))
                ] = key.hex()
        self._touch_paths(
            (self.file_mapper.get_file_name(key) for key in key_list),
            record_policy=False,
        )

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)
        with self._capacity_lock:
            stale_tails = [
                key for key in self._request_group_tails if key[0] == req_context.req_id
            ]
            for key in stale_tails:
                del self._request_group_tails[key]

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()
        now = time()
        with self._capacity_lock:
            if now - self._last_policy_age >= 60.0:
                self._policy.age(now)
                self._last_policy_age = now
            self._flush_metadata_locked()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True

        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
        with self._capacity_lock:
            self._protected_load_paths.clear()
            self._load_paths.clear()
            self._store_evictions.clear()
            self._flush_metadata_locked()
        self._metadata.close()
