"""
Intelligent Caching System with TTL

Features:
- Time-To-Live (TTL) cache
- LRU eviction policy
- Thread-safe operations
- Cache statistics
- Automatic cleanup
- Memory limits

Prevents:
- Redundant API calls
- Slow repeated operations
- Memory exhaustion
- Stale data usage
"""

import time
import threading
from typing import Any, Optional, Dict, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ttl: float = 300.0  # 5 minutes default
    access_count: int = 0
    size_bytes: int = 0

    @property
    def age(self) -> float:
        """Age in seconds"""
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return self.age > self.ttl

    @property
    def time_remaining(self) -> float:
        """Time remaining before expiration (seconds)"""
        return max(0, self.ttl - self.age)


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class TTLCache:
    """
    Thread-safe cache with Time-To-Live and LRU eviction.

    Features:
    - Automatic expiration based on TTL
    - LRU eviction when cache is full
    - Thread-safe operations
    - Statistics tracking
    - Memory management
    - Background cleanup
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 300.0,  # 5 minutes
        max_memory_mb: float = 100.0,
        cleanup_interval: float = 60.0
    ):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            max_memory_mb: Maximum cache memory in MB
            cleanup_interval: Cleanup thread interval in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.cleanup_interval = cleanup_interval

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()

        self._cleanup_thread = None
        self._shutdown = False

        # Start cleanup thread
        self._start_cleanup()

        logger.info(f"TTL Cache initialized: max_size={max_size}, default_ttl={default_ttl}s")

    def _start_cleanup(self):
        """Start background cleanup thread"""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="CacheCleanup"
        )
        self._cleanup_thread.start()

    def _cleanup_worker(self):
        """Background worker that removes expired entries"""
        while not self._shutdown:
            try:
                time.sleep(self.cleanup_interval)
                self._remove_expired()
            except Exception as e:
                logger.error(f"Error in cache cleanup worker: {e}")

    def _remove_expired(self):
        """Remove all expired entries"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]

            for key in expired_keys:
                entry = self._cache.pop(key)
                self._stats.expirations += 1
                self._stats.total_size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1

            if expired_keys:
                logger.debug(f"Removed {len(expired_keys)} expired cache entries")

    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self._cache:
            return

        # OrderedDict maintains insertion order
        # Move accessed items to end, so first item is LRU
        key, entry = self._cache.popitem(last=False)
        self._stats.evictions += 1
        self._stats.total_size_bytes -= entry.size_bytes
        self._stats.entry_count -= 1

        logger.debug(f"Evicted LRU entry: {key} (age: {entry.age:.1f}s, accesses: {entry.access_count})")

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple, dict)):
                # Rough estimate
                return len(str(value))
            else:
                # Generic estimate
                return len(str(value))
        except Exception:
            return 100  # Default estimate

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found or expired

        Returns:
            Cached value or default
        """
        with self._lock:
            entry = self._cache.get(key)

            # Cache miss
            if entry is None:
                self._stats.misses += 1
                return default

            # Expired
            if entry.is_expired:
                self._cache.pop(key)
                self._stats.expirations += 1
                self._stats.misses += 1
                self._stats.total_size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
                return default

            # Cache hit
            self._stats.hits += 1
            entry.access_count += 1
            entry.last_accessed = time.time()

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use default)
        """
        with self._lock:
            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._stats.total_size_bytes -= old_entry.size_bytes
                self._stats.entry_count -= 1

            # Evict if at max size
            while len(self._cache) >= self.max_size:
                self._evict_lru()

            # Evict if exceeding memory limit
            size_bytes = self._estimate_size(value)
            while self._stats.total_size_bytes + size_bytes > self.max_memory_bytes and self._cache:
                self._evict_lru()

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl if ttl is not None else self.default_ttl,
                size_bytes=size_bytes
            )

            self._cache[key] = entry
            self._stats.total_size_bytes += size_bytes
            self._stats.entry_count += 1

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry:
                self._stats.total_size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
            logger.info("Cache cleared")

    def get_or_compute(
        self,
        key: str,
        compute_func: Callable[[], Any],
        ttl: Optional[float] = None
    ) -> Any:
        """
        Get value from cache or compute it.

        If key not in cache or expired, call compute_func and cache result.

        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            ttl: TTL for computed value

        Returns:
            Cached or computed value
        """
        # Try cache first
        cached = self.get(key)
        if cached is not None:
            return cached

        # Compute value
        value = compute_func()

        # Cache it
        self.set(key, value, ttl)

        return value

    def memoize(self, ttl: Optional[float] = None):
        """
        Decorator to memoize function results.

        Usage:
            @cache.memoize(ttl=60)
            def expensive_function(arg1, arg2):
                # ... expensive computation
                return result
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
                cache_key = self._hash_key(key_data)

                # Get or compute
                return self.get_or_compute(
                    cache_key,
                    lambda: func(*args, **kwargs),
                    ttl
                )

            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper

        return decorator

    def _hash_key(self, key_data: Any) -> str:
        """Generate hash key from data"""
        try:
            # Convert to JSON string and hash
            json_str = json.dumps(key_data, sort_keys=True, default=str)
            return hashlib.md5(json_str.encode()).hexdigest()
        except Exception:
            # Fallback to str representation
            return hashlib.md5(str(key_data).encode()).hexdigest()

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                total_size_bytes=self._stats.total_size_bytes,
                entry_count=len(self._cache)
            )

    def get_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get info about cache entry"""
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None

            return {
                'key': key,
                'age': entry.age,
                'ttl': entry.ttl,
                'time_remaining': entry.time_remaining,
                'access_count': entry.access_count,
                'size_bytes': entry.size_bytes,
                'is_expired': entry.is_expired
            }

    def shutdown(self):
        """Shutdown cache and cleanup thread"""
        self._shutdown = True
        self.clear()
        logger.info("Cache shutdown complete")


# Global cache instance
_global_cache: Optional[TTLCache] = None


def get_cache(
    max_size: int = 1000,
    default_ttl: float = 300.0,
    max_memory_mb: float = 100.0
) -> TTLCache:
    """Get or create global cache instance"""
    global _global_cache

    if _global_cache is None:
        _global_cache = TTLCache(
            max_size=max_size,
            default_ttl=default_ttl,
            max_memory_mb=max_memory_mb
        )

    return _global_cache


# Convenience decorator using global cache
def cached(ttl: Optional[float] = None):
    """
    Decorator to cache function results using global cache.

    Usage:
        @cached(ttl=60)
        def expensive_function(arg):
            return expensive_computation(arg)
    """
    cache = get_cache()
    return cache.memoize(ttl=ttl)


if __name__ == '__main__':
    # Test cache
    print("Testing TTL Cache...")
    print("=" * 80)

    cache = TTLCache(max_size=5, default_ttl=2.0)

    # Test basic operations
    cache.set('key1', 'value1')
    cache.set('key2', 'value2', ttl=10.0)

    print(f"✓ Get key1: {cache.get('key1')}")
    print(f"✓ Get key2: {cache.get('key2')}")
    print(f"✓ Get missing: {cache.get('missing', 'default')}")

    # Test expiration
    print("\nWaiting for expiration...")
    time.sleep(2.5)
    print(f"✓ Get expired key1: {cache.get('key1', 'EXPIRED')}")
    print(f"✓ Get key2 (longer TTL): {cache.get('key2')}")

    # Test LRU eviction
    print("\nTesting LRU eviction...")
    for i in range(10):
        cache.set(f'key{i}', f'value{i}')

    stats = cache.get_stats()
    print(f"\nCache Stats:")
    print(f"  Entries: {stats.entry_count}")
    print(f"  Hits: {stats.hits}")
    print(f"  Misses: {stats.misses}")
    print(f"  Hit Rate: {stats.hit_rate:.1f}%")
    print(f"  Evictions: {stats.evictions}")
    print(f"  Expirations: {stats.expirations}")
    print(f"  Size: {stats.total_size_bytes} bytes")

    # Test memoization
    print("\nTesting memoization...")

    @cache.memoize(ttl=5.0)
    def expensive_function(x):
        print(f"  Computing for x={x}...")
        time.sleep(0.1)
        return x * 2

    print(f"First call: {expensive_function(5)}")  # Computes
    print(f"Second call: {expensive_function(5)}")  # Cached
    print(f"Different arg: {expensive_function(10)}")  # Computes

    cache.shutdown()

    print("\n" + "=" * 80)
    print("Cache tests complete!")
