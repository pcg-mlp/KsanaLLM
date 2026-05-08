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

For PD deployments that span multiple **machines** (where shared memory is
insufficient), call :func:`set_request_id_allocator` during startup to plug
in a Redis-backed allocator instead.  When ``None`` is passed, the module
reverts to the default shared-memory counter.
"""

from __future__ import annotations

import fcntl
import logging
import mmap
import os
import struct
from pathlib import Path
from typing import Callable, Optional

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


__all__ = ["SUB_ID_BITS", "set_request_id_allocator", "generate_request_id", "reset_counter_file"]
