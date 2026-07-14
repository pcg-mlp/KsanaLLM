# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""
Comm ID Allocator

Provides multiple implementations for generating globally unique comm_ids:
- Redis: For distributed systems
- SharedMemory: For single-machine multi-process systems using /dev/shm/ with mmap
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import mmap
import os
import struct
from abc import ABC, abstractmethod
from pathlib import Path

import redis
import redis.asyncio as redis_async

logger = logging.getLogger(__name__)

_MAX_COMM_ID = 2147483647  # int32 positive max


class CommIdAllocator(ABC):
    """Base class for comm_id allocation strategies."""

    @abstractmethod
    def allocate_sync(self) -> int:
        """Allocate a unique comm_id synchronously. Returns a positive int32 value."""

    @abstractmethod
    async def allocate(self) -> int:
        """Allocate a unique comm_id asynchronously. Returns a positive int32 value."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""


class RedisAllocator(CommIdAllocator):
    """Allocate comm_ids using Redis (supports distributed systems)."""

    _ALLOC_COMM_ID_LUA = """
local current = redis.call('incr', KEYS[1])
if current > tonumber(ARGV[1]) then
    redis.call('set', KEYS[1], 0)
    return 1
end
return current
"""

    def __init__(self, redis_host: str, redis_port: int, redis_password: str | None, redis_db: int, redis_key: str):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.redis_db = redis_db
        self.redis_key = redis_key
        self.redis_client: redis.Redis | None = None  # Sync client
        self.redis_async_client: redis_async.Redis | None = None  # Async client

    def _get_sync_client(self) -> redis.Redis:
        """Lazy initialize synchronous Redis client."""
        if self.redis_client is None:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                db=self.redis_db,
                decode_responses=False,
            )
        return self.redis_client

    async def _get_async_client(self) -> redis_async.Redis:
        """Lazy initialize asynchronous Redis client."""
        if self.redis_async_client is None:
            self.redis_async_client = redis_async.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                db=self.redis_db,
                decode_responses=False,
            )
        return self.redis_async_client

    def allocate_sync(self) -> int:
        """Allocate comm_id from Redis using Lua script (synchronous)."""
        client = self._get_sync_client()
        try:
            comm_id = client.eval(self._ALLOC_COMM_ID_LUA, 1, self.redis_key, _MAX_COMM_ID)
            return comm_id
        except Exception as exc:
            logger.error("Redis comm_id allocation failed (sync): %s", exc)
            raise

    async def allocate(self) -> int:
        """Allocate comm_id from Redis using Lua script (async)."""
        client = await self._get_async_client()
        try:
            comm_id = await client.eval(self._ALLOC_COMM_ID_LUA, 1, self.redis_key, _MAX_COMM_ID)
            return comm_id
        except Exception as exc:
            logger.error("Redis comm_id allocation failed (async): %s", exc)
            raise

    async def close(self) -> None:
        """Close Redis connections."""
        if self.redis_client is not None:
            try:
                self.redis_client.close()
                self.redis_client = None
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Error closing sync Redis client: %s", exc)
        
        if self.redis_async_client is not None:
            try:
                await self.redis_async_client.aclose()
                self.redis_async_client = None
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Error closing async Redis client: %s", exc)


class SharedMemoryAllocator(CommIdAllocator):
    """Allocate comm_ids using /dev/shm/ with memory-mapped I/O.

    Uses mmap for ultra-fast in-memory operations with fcntl.lockf() for
    inter-process locking.  Each uvicorn worker opens its own fd to the
    shared file; fcntl POSIX locks arbitrate between *processes* correctly.

    Limitation: fcntl locks are per-process, so concurrent threads within
    the same process are NOT mutually excluded.  This is safe with the
    default uvicorn async model (single-threaded event loop per worker).
    If you need true multi-thread safety or distributed deployments,
    use RedisAllocator instead.
    """

    def __init__(self, comm_id_key: str = "default"):
        # Generate filename based on comm_id_key
        self.shm_file = f"/dev/shm/ksana_router_comm_id_{comm_id_key}.bin"
        self._mmap: mmap.mmap | None = None
        self._mmap_fd: int | None = None
        self._ensure_shm_file()

    def _ensure_shm_file(self) -> None:
        """Initialize /dev/shm/ file for mmap."""
        shm_path = Path(self.shm_file)
        
        # Create parent directory if needed
        shm_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file if it doesn't exist
        if not shm_path.exists():
            try:
                with open(self.shm_file, "wb") as f:
                    # Write initial counter value (99) as 4-byte signed integer
                    # First allocation will increment to 100
                    f.write(struct.pack("i", 99))
                logger.info("Created shared memory file: %s", self.shm_file)
            except Exception as exc:
                logger.error("Failed to create shared memory file %s: %s", self.shm_file, exc)
                raise

    def _get_mmap(self) -> mmap.mmap:
        """Get or create mmap object (lazy initialization)."""
        if self._mmap is None:
            try:
                # Open file descriptor
                self._mmap_fd = os.open(self.shm_file, os.O_RDWR)
                # Create mmap for 4 bytes (the integer counter)
                self._mmap = mmap.mmap(self._mmap_fd, 4)
                logger.debug("Initialized mmap for %s", self.shm_file)
            except Exception as exc:
                logger.error("Failed to initialize mmap: %s", exc)
                if self._mmap_fd is not None:
                    os.close(self._mmap_fd)
                raise
        return self._mmap

    async def allocate(self) -> int:
        """Allocate comm_id using mmap (lock-free, ultra-fast).
        
        Uses memory-mapped I/O to avoid system call overhead.
        The 4-byte write on /dev/shm/ is atomic on modern systems.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.allocate_sync)

    def allocate_sync(self) -> int:
        """Synchronous allocation using mmap + fcntl.lockf().

        Safe across multiple processes (each with its own fd).
        NOT safe across multiple threads within the same process — see class docstring.
        """
        try:
            mm = self._get_mmap()
            fd = self._mmap_fd  # Use stored file descriptor instead of mm.fileno()
            
            # Acquire exclusive lock on the 4-byte comm_id counter
            fcntl.lockf(fd, fcntl.LOCK_EX, 4, 0)
            try:
                # Read current value
                mm.seek(0)
                data = mm.read(4)
                current = struct.unpack("i", data)[0]
                
                # Increment and wrap around
                current += 1
                if current > _MAX_COMM_ID:
                    current = 1
                
                # Write back and flush
                mm.seek(0)
                mm.write(struct.pack("i", current))
                mm.flush()  # Ensure visibility to other processes
                
                return current
            finally:
                # Always release the lock
                fcntl.lockf(fd, fcntl.LOCK_UN, 4, 0)
        except Exception as exc:
            logger.error("Shared memory comm_id allocation failed: %s", exc)
            raise

    async def close(self) -> None:
        """Clean up mmap and file descriptor."""
        if self._mmap is not None:
            try:
                self._mmap.close()
                self._mmap = None
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Error closing mmap: %s", exc)
        
        if self._mmap_fd is not None:
            try:
                os.close(self._mmap_fd)
                self._mmap_fd = None
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Error closing mmap fd: %s", exc)


class CommIdAllocatorFactory:
    """Factory for creating comm_id allocators based on configuration."""

    @staticmethod
    def create(
        allocator_type: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: str | None = None,
        redis_db: int = 0,
        redis_key: str = "ksana:comm_id",
    ) -> CommIdAllocator:
        """Create a comm_id allocator.

        Args:
            allocator_type:
                "shared_memory" — fast, single-machine only, safe across
                    uvicorn workers (multi-process) but not multi-threaded
                    callers within the same process.
                "redis" — slightly slower but safe across distributed nodes,
                    multiple processes, and multiple threads.  Preferred for
                    production multi-instance deployments.
            redis_host: Redis server host
            redis_port: Redis server port
            redis_password: Redis server password
            redis_db: Redis database number
            redis_key: Redis key for storing counter (also used as shared_memory filename suffix)

        Returns:
            CommIdAllocator instance

        Raises:
            RuntimeError: If allocator_type is invalid
        """
        if allocator_type == "redis":
            logger.info("Using Redis comm_id allocator (host=%s:%d, db=%d, key=%s)", 
                       redis_host, redis_port, redis_db, redis_key)
            return RedisAllocator(redis_host, redis_port, redis_password, redis_db, redis_key)
        
        elif allocator_type == "shared_memory":
            logger.info("Using SharedMemory comm_id allocator (key=%s)", redis_key)
            return SharedMemoryAllocator(comm_id_key=redis_key)
        
        else:
            raise RuntimeError(f"Unknown allocator type: {allocator_type}. Must be 'redis' or 'shared_memory'.")


# Global allocator instance
_ALLOCATOR: CommIdAllocator | None = None


def init_allocator(
    allocator_type: str = "shared_memory",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_password: str | None = None,
    redis_db: int = 0,
    redis_key: str = "ksana:comm_id",
) -> CommIdAllocator:
    """Initialize the global comm_id allocator.
    
    Should be called once during application startup.
    
    Args:
        allocator_type: "redis" or "shared_memory" (default: "shared_memory")
        redis_key: For redis allocator, the Redis key; for shared_memory, used as filename suffix
    """
    global _ALLOCATOR
    _ALLOCATOR = CommIdAllocatorFactory.create(
        allocator_type=allocator_type,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
        redis_db=redis_db,
        redis_key=redis_key,
    )
    return _ALLOCATOR


def allocate_comm_id() -> int:
    """Allocate a unique comm_id synchronously.

    Typically < 1μs for SharedMemoryAllocator, 1-10ms for RedisAllocator.
    """
    if _ALLOCATOR is None:
        raise RuntimeError("Allocator not initialized. Call init_allocator() first.")
    return _ALLOCATOR.allocate_sync()


async def close_allocator() -> None:
    """Close the global allocator (cleanup on shutdown)."""
    global _ALLOCATOR
    if _ALLOCATOR is not None:
        await _ALLOCATOR.close()
        _ALLOCATOR = None
