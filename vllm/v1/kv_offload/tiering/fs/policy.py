# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission and eviction policies for the filesystem KV tier."""

import hashlib
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Iterable


@dataclass
class CacheEntry:
    """Persistent metadata for one filesystem KV cache entry."""

    cache_key: str
    path: str
    parent_key: str | None = None
    request_prefix_id: str | None = None
    size_bytes: int = 0
    token_count: int = 0
    token_start: int = 0
    token_end: int = 0
    created_at: float = 0.0
    last_access_at: float = 0.0
    access_count: int = 0
    decayed_frequency: float = 0.0
    estimated_prefill_ms: float = 0.0
    observed_load_ms_ema: float = 0.0
    observed_store_ms_ema: float = 0.0
    distinct_session_count: int = 0
    child_count: int = 0
    depth: int = 0
    segment: str = "WINDOW"
    state: str = "READY"


class FrequencySketch:
    """Small Count-Min Sketch used for admission frequency estimates."""

    def __init__(self, width: int = 4096, depth: int = 4) -> None:
        self._width = width
        self._depth = depth
        self._counters = [[0] * width for _ in range(depth)]

    def _index(self, key: str, row: int) -> int:
        digest = hashlib.blake2b(
            f"{row}:{key}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big") % self._width

    def increment(self, key: str) -> None:
        for row in range(self._depth):
            index = self._index(key, row)
            self._counters[row][index] = min(
                self._counters[row][index] + 1, 2**31 - 1
            )

    def estimate(self, key: str) -> int:
        return min(
            self._counters[row][self._index(key, row)]
            for row in range(self._depth)
        )

    def age(self) -> None:
        for row in self._counters:
            for index, value in enumerate(row):
                row[index] = value // 2


class PrefixCostAwareWTinyLFU:
    """W-TinyLFU policy weighted by prefix sharing and prefill cost."""

    WINDOW = "WINDOW"
    PROBATION = "PROBATION"
    PROTECTED = "PROTECTED"

    def __init__(
        self,
        recency_half_life_seconds: float = 3600.0,
        prefix_weight: float = 1.0,
        prefill_tokens_per_second: float = 1000.0,
        frequency_sketch_half_life_seconds: float = 3600.0,
        window_ratio: float = 0.05,
        probation_ratio: float = 0.20,
        protected_ratio: float = 0.75,
    ) -> None:
        if recency_half_life_seconds <= 0:
            raise ValueError("recency_half_life_seconds must be positive")
        if prefix_weight < 0:
            raise ValueError("prefix_weight must be non-negative")
        if prefill_tokens_per_second <= 0:
            raise ValueError("prefill_tokens_per_second must be positive")
        if frequency_sketch_half_life_seconds <= 0:
            raise ValueError("frequency_sketch_half_life_seconds must be positive")
        ratios = (window_ratio, probation_ratio, protected_ratio)
        if any(ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
            raise ValueError("cache segment ratios must be non-negative and sum to 1")
        self.recency_half_life_seconds = recency_half_life_seconds
        self.prefix_weight = prefix_weight
        self.prefill_tokens_per_second = prefill_tokens_per_second
        self.frequency_sketch_half_life_seconds = (
            frequency_sketch_half_life_seconds
        )
        self.segment_ratios = {
            self.WINDOW: window_ratio,
            self.PROBATION: probation_ratio,
            self.PROTECTED: protected_ratio,
        }
        self.sketch = FrequencySketch()
        self._last_sketch_age = time.time()
        self.entries: dict[str, CacheEntry] = {}
        self._sessions: dict[str, set[str]] = {}

    def load(self, entries: Iterable[CacheEntry]) -> None:
        """Restore metadata and seed the frequency sketch after a restart."""
        for entry in entries:
            self.entries[entry.cache_key] = entry
            for _ in range(min(max(entry.access_count, 0), 32)):
                self.sketch.increment(entry.cache_key)

    def _effective_frequency(self, entry: CacheEntry, now: float) -> float:
        if entry.last_access_at <= 0:
            return max(entry.decayed_frequency, 0.0)
        idle = max(now - entry.last_access_at, 0.0)
        return max(entry.decayed_frequency, 0.0) * 0.5 ** (
            idle / self.recency_half_life_seconds
        )

    def record_access(
        self,
        cache_key: str,
        now: float | None = None,
        session_id: str | None = None,
    ) -> None:
        """Record a lookup or explicit touch without synchronous DB I/O."""
        now = time.time() if now is None else now
        self.sketch.increment(cache_key)
        entry = self.entries.get(cache_key)
        if entry is None:
            return
        effective_frequency = self._effective_frequency(entry, now)
        entry.last_access_at = now
        entry.access_count += 1
        entry.decayed_frequency = effective_frequency + 1.0
        if session_id is not None:
            sessions = self._sessions.setdefault(cache_key, set())
            sessions.add(session_id)
            entry.distinct_session_count = max(
                entry.distinct_session_count, len(sessions)
            )
        if entry.segment == self.WINDOW and entry.access_count > 1:
            entry.segment = self.PROBATION
        elif entry.segment == self.PROBATION:
            entry.segment = self.PROTECTED

    def on_store(
        self,
        entry: CacheEntry,
        now: float | None = None,
    ) -> None:
        """Add a successfully written entry to the W-TinyLFU window."""
        now = time.time() if now is None else now
        previous = self.entries.get(entry.cache_key)
        if previous is not None:
            entry.access_count = previous.access_count
            entry.decayed_frequency = previous.decayed_frequency
            entry.distinct_session_count = previous.distinct_session_count
            entry.child_count = previous.child_count
            if entry.observed_load_ms_ema <= 0:
                entry.observed_load_ms_ema = previous.observed_load_ms_ema
            if entry.observed_store_ms_ema <= 0:
                entry.observed_store_ms_ema = previous.observed_store_ms_ema
            entry.segment = previous.segment
            entry.created_at = previous.created_at
            if previous.parent_key != entry.parent_key:
                if previous.parent_key is not None:
                    parent = self.entries.get(previous.parent_key)
                    if parent is not None:
                        parent.child_count = max(parent.child_count - 1, 0)
                if entry.parent_key is not None:
                    parent = self.entries.get(entry.parent_key)
                    if parent is not None:
                        parent.child_count += 1
        elif entry.parent_key is not None:
            parent = self.entries.get(entry.parent_key)
            if parent is not None:
                parent.child_count += 1
        if entry.created_at <= 0:
            entry.created_at = now
        entry.last_access_at = max(entry.last_access_at, now)
        entry.state = "READY"
        self.entries[entry.cache_key] = entry
        self.sketch.increment(entry.cache_key)

    def on_remove(self, cache_key: str) -> CacheEntry | None:
        """Remove an entry from the policy after its file is deleted."""
        entry = self.entries.pop(cache_key, None)
        self._sessions.pop(cache_key, None)
        if entry is not None and entry.parent_key is not None:
            parent = self.entries.get(entry.parent_key)
            if parent is not None:
                parent.child_count = max(parent.child_count - 1, 0)
        for child in self.entries.values():
            if child.parent_key == cache_key:
                child.parent_key = None
                child.depth = 0
                child.token_start = 0
                child.token_end = child.token_count
        return entry

    def age(self, now: float | None = None) -> None:
        """Decay frequency and move stale protected entries to probation."""
        now = time.time() if now is None else now
        for entry in self.entries.values():
            idle = max(now - entry.last_access_at, 0.0)
            if (
                entry.segment == self.PROTECTED
                and idle > 2 * self.recency_half_life_seconds
            ):
                entry.segment = self.PROBATION
        if now - self._last_sketch_age >= self.frequency_sketch_half_life_seconds:
            self.sketch.age()
            self._last_sketch_age = now

    def make_entry(
        self,
        cache_key: str,
        path: str,
        size_bytes: int,
        token_count: int,
        parent_key: str | None = None,
        session_id: str | None = None,
    ) -> CacheEntry:
        """Create candidate metadata using the configured prefill estimate."""
        depth = 0
        token_start = 0
        if parent_key is not None:
            parent = self.entries.get(parent_key)
            depth = 1 if parent is None else parent.depth + 1
            token_start = 0 if parent is None else parent.token_end
        return CacheEntry(
            cache_key=cache_key,
            path=path,
            parent_key=parent_key,
            request_prefix_id=session_id,
            size_bytes=size_bytes,
            token_count=token_count,
            token_start=token_start,
            token_end=token_start + token_count,
            created_at=time.time(),
            last_access_at=time.time(),
            decayed_frequency=float(self.sketch.estimate(cache_key)),
            estimated_prefill_ms=(
                token_count / self.prefill_tokens_per_second * 1000.0
            ),
            depth=depth,
        )

    def score(self, entry: CacheEntry, now: float | None = None) -> float:
        """Return the retained value per byte; lower values evict first."""
        now = time.time() if now is None else now
        idle = max(now - entry.last_access_at, 0.0)
        recency = math.exp(-idle / self.recency_half_life_seconds)
        frequency = self._effective_frequency(entry, now) + 1.0
        shared_count = max(entry.distinct_session_count, 0) + max(
            entry.child_count, 0
        )
        shared_bonus = 1.0 + math.log2(1.0 + shared_count)
        compute_value = max(
            entry.estimated_prefill_ms - entry.observed_load_ms_ema, 0.0
        )
        prefix_bonus = 1.0 + self.prefix_weight / (1.0 + entry.depth)
        one_hit_factor = 0.5 if entry.access_count <= 1 else 1.0
        return (
            frequency
            * (0.25 + 0.75 * recency)
            * shared_bonus
            * prefix_bonus
            * compute_value
            * one_hit_factor
            / max(entry.size_bytes, 1)
        )

    def should_admit(
        self,
        candidate: CacheEntry,
        victim_candidates: list[CacheEntry],
    ) -> bool:
        """Reject one-hit pollution when a victim has equal frequency."""
        if not victim_candidates:
            return True
        victim = min(victim_candidates, key=self.score)
        candidate_frequency = self.sketch.estimate(candidate.cache_key)
        victim_frequency = self.sketch.estimate(victim.cache_key)
        if candidate_frequency != victim_frequency:
            return candidate_frequency > victim_frequency
        return self.score(candidate) > self.score(victim) * 2.0

    def select_victims(
        self,
        entries: list[CacheEntry],
        bytes_to_free: int,
        now: float | None = None,
    ) -> list[CacheEntry]:
        """Select low-value READY entries until the requested space is free."""
        now = time.time() if now is None else now
        candidates = [entry for entry in entries if entry.state == "READY"]
        total_bytes = sum(entry.size_bytes for entry in candidates)
        segment_bytes = {
            segment: sum(
                entry.size_bytes for entry in candidates if entry.segment == segment
            )
            for segment in self.segment_ratios
        }
        excess = {
            segment: max(
                segment_bytes[segment] - total_bytes * self.segment_ratios[segment],
                0,
            )
            for segment in self.segment_ratios
        }

        def victim_key(entry: CacheEntry) -> tuple[int, float, str]:
            if entry.segment == self.WINDOW and excess[self.WINDOW] > 0:
                priority = 0
            elif entry.segment == self.PROBATION and excess[self.PROBATION] > 0:
                priority = 1
            elif entry.segment == self.PROTECTED and excess[self.PROTECTED] > 0:
                priority = 2
            else:
                priority = 3
            return priority, self.score(entry, now), entry.path

        candidates.sort(key=victim_key)
        victims: list[CacheEntry] = []
        freed = 0
        for entry in candidates:
            victims.append(entry)
            freed += entry.size_bytes
            if freed >= bytes_to_free:
                break
        return victims


class CacheMetadataStore:
    """SQLite persistence for policy metadata, using WAL and batched writes."""

    def __init__(self, path: str) -> None:
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            path, timeout=30, check_same_thread=False
        )
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    cache_key TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    parent_key TEXT,
                    request_prefix_id TEXT,
                    size_bytes INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    token_start INTEGER NOT NULL,
                    token_end INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    last_access_at REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    decayed_frequency REAL NOT NULL,
                    estimated_prefill_ms REAL NOT NULL,
                    observed_load_ms_ema REAL NOT NULL,
                    observed_store_ms_ema REAL NOT NULL,
                    distinct_session_count INTEGER NOT NULL,
                    child_count INTEGER NOT NULL,
                    depth INTEGER NOT NULL,
                    segment TEXT NOT NULL,
                    state TEXT NOT NULL
                )
                """
            )
            self._connection.commit()

    def load(self) -> list[CacheEntry]:
        """Load all persisted entries."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM entries"
            ).fetchall()
        return [CacheEntry(*row) for row in rows]

    def save(self, entries: Iterable[CacheEntry]) -> None:
        """Batch-save current metadata in one transaction."""
        values = [
            (
                entry.cache_key,
                entry.path,
                entry.parent_key,
                entry.request_prefix_id,
                entry.size_bytes,
                entry.token_count,
                entry.token_start,
                entry.token_end,
                entry.created_at,
                entry.last_access_at,
                entry.access_count,
                entry.decayed_frequency,
                entry.estimated_prefill_ms,
                entry.observed_load_ms_ema,
                entry.observed_store_ms_ema,
                entry.distinct_session_count,
                entry.child_count,
                entry.depth,
                entry.segment,
                entry.state,
            )
            for entry in entries
        ]
        with self._lock:
            self._connection.executemany(
                """
                INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    path=excluded.path, parent_key=excluded.parent_key,
                    request_prefix_id=excluded.request_prefix_id,
                    size_bytes=excluded.size_bytes, token_count=excluded.token_count,
                    token_start=excluded.token_start, token_end=excluded.token_end,
                    created_at=excluded.created_at,
                    last_access_at=excluded.last_access_at,
                    access_count=excluded.access_count,
                    decayed_frequency=excluded.decayed_frequency,
                    estimated_prefill_ms=excluded.estimated_prefill_ms,
                    observed_load_ms_ema=excluded.observed_load_ms_ema,
                    observed_store_ms_ema=excluded.observed_store_ms_ema,
                    distinct_session_count=excluded.distinct_session_count,
                    child_count=excluded.child_count, depth=excluded.depth,
                    segment=excluded.segment, state=excluded.state
                """,
                values,
            )
            self._connection.commit()

    def delete(self, cache_keys: Iterable[str]) -> None:
        """Batch-delete metadata for removed files."""
        with self._lock:
            self._connection.executemany(
                "DELETE FROM entries WHERE cache_key = ?",
                ((cache_key,) for cache_key in cache_keys),
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()
