# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""Unified request ID generation for KsanaLLM Python entry points.

This module is the single source of truth on the Python side for producing the
``request-id`` HTTP header that gets forwarded to the C++ engine.  The C++
``Request`` constructor (``csrc/utils/request.cpp``) prefers the header value
over its internal ``IdGenerator``, so by always generating the ID here we keep
the Python access log and the C++ engine log talking about the same int64 ID.

Bit layout (must stay in sync with ``ksana_llm::IdGenerator`` in C++):

    bit 63        bit 8                 bit 0
      |             |                     |
      v             v                     v
      +---+----------------------+--------+
      | 0 |    base counter      | branch |
      +---+----------------------+--------+
                                 ^
                                 |
                       SUB_ID_BITS = 8 (reserved)

The low ``SUB_ID_BITS`` bits are reserved for the InferRequest branch index
(beam search / num_return_sequences); the Python side always emits
``base << SUB_ID_BITS`` and leaves the branch bits zero.

Default implementation — shared-memory atomic counter
------------------------------------------------------
Both the non-PD path (``serving_server.py``, single-process uvicorn) and the
PD path (``simple_router``, multi-process uvicorn) use the **same**
mmap + fcntl shared-memory counter stored in
``/tmp/ksana_request_id_<key>.bin``.  Each process opens its own file
descriptor; ``fcntl.lockf`` arbitrates between processes.

Multi-node PD deployments — Redis-backed allocator
--------------------------------------------------
When pd_v2 deploys multiple Decode nodes (different machines / containers),
the local shared-memory counter is insufficient: every Decode node would
seed at ``_INITIAL_COUNTER_VALUE = 100_000`` and emit colliding IDs that
break Prefill's per-req routing (``in_flight_`` keyed by uint64 req_id).

Use :func:`install_redis_allocator` (or
:func:`try_install_redis_allocator_from_yaml`) at service startup to plug in
a Redis ``INCR``-backed allocator. All Decode nodes that share the same
``cluster_name`` and Redis instance then draw IDs from a single atomic
counter — guaranteeing cluster-wide uniqueness.
"""

from __future__ import annotations

import fcntl
import logging
import mmap
import os
import struct
from pathlib import Path
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

# Reserve the lowest SUB_ID_BITS bits for the InferRequest branch index.
# Must match ``ksana_llm::IdGenerator::kSubIdBits`` in csrc/utils/id_generator.h.
SUB_ID_BITS: int = 8

# Mask the counter to 63 - SUB_ID_BITS bits so the resulting int64 stays
# non-negative (sign bit kept at 0), matching the C++ IdGenerator behaviour.
_MAX_BASE_COUNTER: int = (1 << (63 - SUB_ID_BITS)) - 1

# Counter wraps at int32 positive max (matches comm_id_allocator behaviour).
# Values stay well within _MAX_BASE_COUNTER.
_MAX_COUNTER_VALUE: int = 2_147_483_647  # int32 positive max

_COUNTER_BYTES: int = 8  # int64 on disk
_COUNTER_FMT: str = "q"  # struct format for int64

# Initial counter value to avoid colliding with C++ IdGenerator-generated IDs used
# during engine warmup. C++ `IdGenerator::Gen()` starts at 0 and produces shifted
# IDs (0, 256, 512, ...), so warmup consumes the low range. Router counter must
# start high enough that (counter << SUB_ID_BITS) never overlaps that range.
# 100000 reserves room for ~100k warmup iterations, far above any realistic budget.
_INITIAL_COUNTER_VALUE: int = 100_000

_SHM_DIR: str = "/dev/shm"
_DEFAULT_SHM_KEY: str = "default"


class _SharedMemoryCounter:
    """Single-file mmap + fcntl atomic counter shared across processes.

    Each process keeps its own fd; ``fcntl.lockf`` provides mutual exclusion
    across processes.  Within a single process the uvicorn async model uses
    one event loop per worker, so single-thread-per-process holds and no
    additional locking is needed.

    Multi-worker safety: every worker's constructor must land on the **same**
    inode, otherwise ``fcntl.lockf`` coordinates nothing and each worker
    generates duplicate IDs. We therefore only create the file if it's
    missing — no unlink/recreate inside the per-worker init. The counter
    persists across service restarts (harmless; just keeps incrementing),
    and :func:`reset_counter_file` is available for tests/manual resets but
    MUST NOT be called from within a worker process.
    """

    def __init__(self, key: str = _DEFAULT_SHM_KEY) -> None:
        self._path = os.path.join(_SHM_DIR, f"ksana_request_id_{key}.bin")
        self._mm: mmap.mmap | None = None
        self._fd: int | None = None
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create the shared counter file if and only if it doesn't exist.

        Uses O_CREAT|O_EXCL so that exactly one worker wins the race and
        writes the initial seed; losers see ``FileExistsError`` and simply
        reuse the existing file. Never unlinks — all workers must share the
        same inode for fcntl locks to coordinate.
        """
        try:
            fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, struct.pack(_COUNTER_FMT, _INITIAL_COUNTER_VALUE))
                logger.debug("Created request-id counter file: %s (seed=%d)",
                             self._path, _INITIAL_COUNTER_VALUE)
            finally:
                os.close(fd)
        except FileExistsError:
            # Another worker (or a previous run) already created the file.
            # We intentionally do not reset — counter persists across runs.
            logger.debug("Reusing existing request-id counter file: %s", self._path)
        except OSError as exc:
            logger.error("Failed to ensure counter file %s: %s", self._path, exc)
            raise

    def _get_mmap(self) -> mmap.mmap:
        if self._mm is None:
            self._fd = os.open(self._path, os.O_RDWR)
            self._mm = mmap.mmap(self._fd, _COUNTER_BYTES)
        return self._mm

    def next(self) -> int:
        """Return the next counter value (wraps at ``_MAX_COUNTER_VALUE``)."""
        mm = self._get_mmap()
        fd = self._fd
        fcntl.lockf(fd, fcntl.LOCK_EX, _COUNTER_BYTES, 0)
        try:
            mm.seek(0)
            current = struct.unpack(_COUNTER_FMT, mm.read(_COUNTER_BYTES))[0]
            current += 1
            if current > _MAX_COUNTER_VALUE:
                current = 1
            mm.seek(0)
            mm.write(struct.pack(_COUNTER_FMT, current))
            mm.flush()
            return current
        finally:
            fcntl.lockf(fd, fcntl.LOCK_UN, _COUNTER_BYTES, 0)

    def close(self) -> None:
        if self._mm is not None:
            try:
                self._mm.close()
            except OSError:
                pass
            self._mm = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


# Module-level default counter — lazily initialised on first use.
_DEFAULT_COUNTER: _SharedMemoryCounter | None = None

# Optional override (e.g. Redis allocator for multi-machine PD deployments).
# The callable must return a fresh positive int (raw counter, before shifting).
_CUSTOM_ALLOCATOR: Optional[Callable[[], int]] = None


def _get_default_counter() -> _SharedMemoryCounter:
    global _DEFAULT_COUNTER
    if _DEFAULT_COUNTER is None:
        _DEFAULT_COUNTER = _SharedMemoryCounter()
    return _DEFAULT_COUNTER


def reset_counter_file(key: str = _DEFAULT_SHM_KEY) -> None:
    """Forcibly reset the shared counter file to the seed value.

    This MUST be called from the parent/launcher process BEFORE any uvicorn
    workers are spawned (e.g. at the top of ``main.py:main()`` before
    ``uvicorn.run(..., workers=N)``). Calling it from inside a worker is
    unsafe: it would unlink the inode shared with other workers and
    desynchronise the counter.

    Args:
        key: The same key used by ``_SharedMemoryCounter``.
    """
    path = os.path.join(_SHM_DIR, f"ksana_request_id_{key}.bin")
    try:
        if Path(path).exists():
            os.unlink(path)
        with open(path, "wb") as f:
            f.write(struct.pack(_COUNTER_FMT, _INITIAL_COUNTER_VALUE))
        logger.info("Reset request-id counter file %s (seed=%d)", path, _INITIAL_COUNTER_VALUE)
    except OSError as exc:
        logger.error("Failed to reset counter file %s: %s", path, exc)
        raise


def set_request_id_allocator(allocator: Optional[Callable[[], int]]) -> None:
    """Override the default shared-memory counter with a custom allocator.

    Use this only when the default shared-memory counter is insufficient,
    e.g. Redis-backed IDs across multiple physical machines.

    Args:
        allocator: A callable returning a fresh positive int (raw counter
            value, before the ``SUB_ID_BITS`` shift).  Pass ``None`` to
            revert to the default shared-memory counter.
    """
    global _CUSTOM_ALLOCATOR
    _CUSTOM_ALLOCATOR = allocator


def generate_request_id() -> int:
    """Generate a unique request ID matching the C++ ``IdGenerator`` layout.

    Uses the shared-memory counter by default so the ID is unique across
    all uvicorn worker processes on the same machine — regardless of whether
    the service is running in PD-separated or standalone mode.

    Returns:
        A non-negative int64 with the lowest ``SUB_ID_BITS`` bits set to zero
        (reserved for the InferRequest branch index in C++).
    """
    if _CUSTOM_ALLOCATOR is not None:
        base = _CUSTOM_ALLOCATOR()
    else:
        base = _get_default_counter().next()
    return (base & _MAX_BASE_COUNTER) << SUB_ID_BITS


__all__ = ["SUB_ID_BITS", "set_request_id_allocator", "generate_request_id",
           "reset_counter_file", "install_redis_allocator",
           "try_install_redis_allocator_from_yaml", "RedisRequestIdAllocator"]


# --------------------------------------------------------------------------- #
# Redis-backed allocator for multi-node PD deployments.
# --------------------------------------------------------------------------- #

# Default Redis key — must include the cluster_name so two unrelated pd_v2
# clusters sharing the same Redis instance don't collide on the counter.
_REDIS_KEY_TEMPLATE: str = "pd_v2/cluster/{cluster_name}/request_id_counter"

# Seed value the FIRST node to touch the key writes via SETNX. Same rationale
# as _INITIAL_COUNTER_VALUE (avoid colliding with C++ warmup IDs).
_REDIS_INITIAL_VALUE: int = _INITIAL_COUNTER_VALUE


class RedisRequestIdAllocator:
    """Cluster-wide unique request-id allocator backed by Redis ``INCR``.

    Use this in multi-node PD setups where every Decode node would otherwise
    seed its own local counter and emit colliding IDs. All nodes that share
    the same ``cluster_name`` and Redis instance draw IDs from a single
    atomic counter — guaranteed unique across the cluster.

    The allocator caches a small batch of IDs per call to amortize the
    network round-trip: each ``INCRBY batch_size`` reserves a contiguous
    range and ``__call__`` hands them out locally. Per-process ``threading``
    locks make concurrent uvicorn workers race-safe.

    On a Redis outage the allocator raises (no silent fallback to the local
    counter — that would break uniqueness, which is the whole point of this
    class).
    """

    def __init__(self, host: str, port: int, *, password: Optional[str] = None,
                 db: int = 0, cluster_name: str = "default",
                 batch_size: int = 64, socket_timeout: float = 2.0) -> None:
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover - dep check
            raise ImportError(
                "RedisRequestIdAllocator requires the 'redis' Python package. "
                "Install via: pip install redis"
            ) from exc

        import threading

        self._client = redis.Redis(
            host=host, port=port, password=password, db=db,
            socket_timeout=socket_timeout, socket_connect_timeout=socket_timeout,
            decode_responses=False,
        )
        self._key = _REDIS_KEY_TEMPLATE.format(cluster_name=cluster_name)
        self._batch_size = max(1, int(batch_size))
        self._lock = threading.Lock()
        # Per-process reservation window. _reserved_high is the largest id we
        # own; _next is the next one to hand out. Empty window = need refill.
        self._next: int = 0
        self._reserved_high: int = -1
        # Seed the counter to _REDIS_INITIAL_VALUE iff this is the first writer.
        # SETNX is atomic, so concurrent boots converge on one seed.
        try:
            self._client.setnx(self._key, _REDIS_INITIAL_VALUE)
        except Exception as exc:
            logger.error("RedisRequestIdAllocator: SETNX seed failed (key=%s): %s",
                         self._key, exc)
            raise
        logger.info("RedisRequestIdAllocator initialized: host=%s:%d key=%s batch=%d",
                    host, port, self._key, self._batch_size)

    def _refill_locked(self) -> None:
        """Reserve a fresh ``batch_size`` window via ``INCRBY``. Holds ``_lock``."""
        high = int(self._client.incrby(self._key, self._batch_size))
        # high is the new counter value AFTER the increment, so the reserved
        # range is (high - batch_size, high] (inclusive on the high end).
        self._next = high - self._batch_size + 1
        self._reserved_high = high

    def __call__(self) -> int:
        with self._lock:
            if self._next > self._reserved_high:
                self._refill_locked()
            val = self._next
            self._next += 1
            return val


def install_redis_allocator(host: str, port: int, *,
                            password: Optional[str] = None, db: int = 0,
                            cluster_name: str = "default",
                            batch_size: int = 64) -> RedisRequestIdAllocator:
    """Install a Redis-backed allocator as the module-level override.

    Subsequent :func:`generate_request_id` calls return cluster-unique IDs
    drawn from ``INCR pd_v2/cluster/<cluster_name>/request_id_counter``.

    Returns the allocator instance so callers can call ``__call__`` directly
    in tests if they want to skip the module-level indirection.
    """
    allocator = RedisRequestIdAllocator(
        host=host, port=port, password=password, db=db,
        cluster_name=cluster_name, batch_size=batch_size,
    )
    set_request_id_allocator(allocator)
    return allocator


def _parse_redis_url(url: str) -> Optional[Tuple[str, int, int]]:
    """Parse ``redis://[:password@]host:port[/db]`` into ``(host, port, db)``.

    The password embedded in the URL is intentionally ignored — the
    deployment convention is to pass it via the ``redis_password`` YAML
    field (mirrors Mooncake's ``MC_REDIS_PASSWORD`` env). Returns ``None``
    if the URL is not a Redis URL we recognize.
    """
    if not isinstance(url, str):
        return None
    if not url.startswith("redis://") and not url.startswith("rediss://"):
        return None
    rest = url.split("://", 1)[1]
    if "@" in rest:
        rest = rest.rsplit("@", 1)[1]  # drop user:pass@ prefix
    db = 0
    if "/" in rest:
        rest, db_part = rest.rsplit("/", 1)
        try:
            db = int(db_part)
        except ValueError:
            db = 0
    if ":" not in rest:
        return None
    host, port_str = rest.rsplit(":", 1)
    try:
        port = int(port_str)
    except ValueError:
        return None
    return (host, port, db)


def try_install_redis_allocator_from_yaml(config_path: str) -> bool:
    """Read the engine YAML and install a Redis allocator iff it's pd_v2.

    Looks for:
      ``setting.connector.backend == "v2"`` (pd_v2 enabled)
      ``setting.connector.metadata_addr`` starts with ``redis://`` (Redis backend)
      ``setting.connector.cluster_name`` (cluster scope for the counter key)
      ``setting.connector.redis_password`` (optional)

    Returns True if the allocator was installed, False otherwise (in which
    case the default shared-memory counter remains active). Logs the
    decision either way so multi-node deployments can verify the wiring.

    Importing yaml is deferred so non-pd_v2 deployments without PyYAML
    installed still work.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.debug(
            "try_install_redis_allocator_from_yaml: PyYAML not available; "
            "leaving default shared-memory counter")
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "try_install_redis_allocator_from_yaml: failed to read %s: %s; "
            "leaving default shared-memory counter", config_path, exc)
        return False

    connector = ((cfg.get("setting") or {}).get("connector") or {})
    backend = connector.get("backend")
    metadata_addr = connector.get("metadata_addr") or ""
    cluster_name = connector.get("cluster_name") or "default"
    password = connector.get("redis_password") or None

    if backend != "v2":
        logger.debug(
            "try_install_redis_allocator_from_yaml: backend=%r != 'v2'; "
            "leaving default shared-memory counter", backend)
        return False

    parsed = _parse_redis_url(metadata_addr)
    if parsed is None:
        logger.info(
            "try_install_redis_allocator_from_yaml: pd_v2 enabled but "
            "metadata_addr=%r is not a redis:// URL; leaving default "
            "shared-memory counter (single-host P2PHANDSHAKE deployment)",
            metadata_addr)
        return False

    host, port, db = parsed
    install_redis_allocator(
        host=host, port=port, password=password, db=db,
        cluster_name=cluster_name,
    )
    logger.info(
        "try_install_redis_allocator_from_yaml: installed Redis allocator "
        "host=%s:%d db=%d cluster=%s (request-id space is now cluster-wide unique)",
        host, port, db, cluster_name)
    return True
