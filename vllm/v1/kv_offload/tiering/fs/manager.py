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
from time import time
from typing import TYPE_CHECKING, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
    make_offload_key,
    get_offload_group_idx,
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
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool
from vllm.v1.kv_offload.tiering.fs.policy import (
    CacheEntry,
    CacheMetadataStore,
    PrefixCostAwareWTinyLFU,
)

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
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        if max_bytes is not None and (
            isinstance(max_bytes, bool) or not isinstance(max_bytes, int)
        ):
            raise TypeError("max_bytes must be a non-negative integer or None")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer or None")
        if cache_policy != "prefix_cost_aware_wtinylfu":
            raise ValueError(
                "cache_policy must be 'prefix_cost_aware_wtinylfu'"
            )
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
        )
        self._storage_dir = f"{self.file_mapper.base_path}_r{self.file_mapper.rank}"
        self._capacity_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._entries: dict[str, int] = {}
        self._path_to_key: dict[str, OffloadKey] = {}
        self._protected_load_paths: set[str] = set()
        self._policy = PrefixCostAwareWTinyLFU(
            recency_half_life_seconds=recency_half_life_seconds,
            prefix_weight=prefix_weight,
            prefill_tokens_per_second=prefill_tokens_per_second,
        )
        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        self._metadata = CacheMetadataStore(
            f"{self._storage_dir}.metadata.sqlite3"
        )
        self._policy.load(self._metadata.load())

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
        self._policy.record_access(
            key.hex(), session_id=req_context.req_id
        )
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
        stale_keys = [
            key for key in self._policy.entries if key not in current_keys
        ]
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
                entry.state = "READY"
            self._entries[path] = size
            self._path_to_key[path] = key

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
            return key
        except OSError:
            return None
        self._entries.pop(path, None)
        key = self._path_to_key.pop(path, None)
        if key is not None:
            self._policy.on_remove(key.hex())
        return key

    def _token_count(self, key: OffloadKey) -> int:
        """Return the token span represented by one persisted chunk."""
        group_idx = get_offload_group_idx(key)
        tokens_per_block = getattr(self._offloading_spec, "tokens_per_block", ())
        if group_idx < len(tokens_per_block):
            return int(tokens_per_block[group_idx]) * self._offloading_spec.blocks_per_chunk
        return int(self._offloading_spec.tokens_per_hash) * self._offloading_spec.blocks_per_chunk

    def _make_room_locked(
        self, paths: list[str], session_id: str | None = None
    ) -> tuple[bool, list[OffloadKey]]:
        """Reserve space using frequency, prefix, cost and recency value."""
        if self.max_bytes is None:
            return True, []

        self._refresh_entries_locked()
        missing_paths = list(dict.fromkeys(p for p in paths if not os.path.exists(p)))
        required_bytes = len(missing_paths) * self._block_size
        if required_bytes > self.max_bytes:
            return False, []

        current_bytes = sum(self._entries.values())
        if current_bytes + required_bytes <= self.max_bytes:
            return True, []

        protected = self._protected_load_paths | set(paths)
        entries = [
            self._policy.entries[key.hex()]
            for path, key in self._path_to_key.items()
            if path not in protected
            and key.hex() in self._policy.entries
            and self._policy.entries[key.hex()].state == "READY"
        ]
        bytes_to_free = current_bytes + required_bytes - self.max_bytes
        victims = self._policy.select_victims(entries, bytes_to_free)
        if sum(entry.size_bytes for entry in victims) < bytes_to_free:
            return False, []

        victim_candidates = list(victims)
        for path in missing_paths:
            key = self._path_to_key.get(path) or self._parse_key(path)
            if key is None:
                continue
            candidate = self._policy.make_entry(
                cache_key=key.hex(),
                path=path,
                size_bytes=self._block_size,
                token_count=self._token_count(key),
                session_id=session_id,
            )
            if not self._policy.should_admit(candidate, victim_candidates):
                return False, []

        evicted: list[OffloadKey] = []
        for entry in victims:
            key = self._remove_entry_locked(entry.path)
            if key is not None:
                evicted.append(key)
        return True, evicted

    def _store_batch(
        self,
        job_id: JobId,
        paths: list[str],
        offsets: list[int],
        session_id: str | None,
    ) -> None:
        """Admit and write one batch while serializing capacity decisions."""
        with self._capacity_lock:
            admitted, evicted = self._make_room_locked(paths, session_id)
            if self.events is not None:
                self._store_evictions[job_id] = evicted
            if not admitted:
                logger.warning_once(
                    "Skipping filesystem KV cache store for job %s: the batch "
                    "does not fit in max_bytes=%d.",
                    job_id,
                    self.max_bytes,
                )
                return
            try:
                batch_store_block(
                    paths,
                    self._primary_kv_view,
                    offsets,
                    self._block_size,
                    self._use_o_direct,
                )
                previous_key: str | None = None
                for path in paths:
                    key = self._parse_key(path)
                    if key is None or not os.path.exists(path):
                        continue
                    entry = self._policy.make_entry(
                        cache_key=key.hex(),
                        path=path,
                        size_bytes=os.path.getsize(path),
                        token_count=self._token_count(key),
                        parent_key=previous_key,
                        session_id=session_id,
                    )
                    self._policy.on_store(entry)
                    previous_key = key.hex()
            finally:
                self._refresh_entries_locked()
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
                            sample_ms = transfer_time * 1000 / max(
                                len(load_paths), 1
                            )
                            entry.observed_load_ms_ema = (
                                0.2 * sample_ms
                                + 0.8 * entry.observed_load_ms_ema
                            )
                    else:
                        self._policy.on_remove(key.hex())
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
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        key_list = list(keys)
        for key in key_list:
            self._policy.record_access(key.hex(), session_id=req_context.req_id)
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
