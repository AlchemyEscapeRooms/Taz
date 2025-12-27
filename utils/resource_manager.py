"""
Resource Management System

Provides:
- Database connection pooling
- API key validation and management
- Resource lifecycle management
- Connection health monitoring
- Automatic cleanup and recovery

Prevents:
- Connection leaks
- Resource exhaustion
- Invalid API credentials
- Zombie connections
"""

import os
import sqlite3
import threading
import time
import queue
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .exceptions import (
    ConnectionPoolExhaustedError,
    ResourceExhaustedError,
    InvalidConfigError,
    APIAuthenticationError,
    ConfigurationError
)

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for connection pool monitoring"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_wait_time: float = 0.0
    max_wait_time: float = 0.0
    pool_exhaustions: int = 0


@dataclass
class PooledConnection:
    """Wrapper for pooled database connection"""
    connection: sqlite3.Connection
    in_use: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    thread_id: Optional[int] = None

    @property
    def age_seconds(self) -> float:
        """Age of connection in seconds"""
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def idle_seconds(self) -> float:
        """Time since last use in seconds"""
        return (datetime.now() - self.last_used).total_seconds()


class DatabaseConnectionPool:
    """
    Thread-safe database connection pool.

    Features:
    - Connection reuse
    - Automatic cleanup of idle connections
    - Health checks
    - Statistics tracking
    - Resource limits
    """

    def __init__(
        self,
        database_path: str,
        pool_size: int = 10,
        max_overflow: int = 5,
        timeout: float = 30.0,
        max_connection_age: int = 3600,  # 1 hour
        max_idle_time: int = 300  # 5 minutes
    ):
        """
        Initialize connection pool.

        Args:
            database_path: Path to SQLite database file
            pool_size: Core pool size
            max_overflow: Maximum connections beyond pool_size
            timeout: Timeout waiting for connection (seconds)
            max_connection_age: Maximum connection age (seconds)
            max_idle_time: Maximum idle time before recycling (seconds)
        """
        self.database_path = database_path
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        self.max_connection_age = max_connection_age
        self.max_idle_time = max_idle_time

        self._pool: List[PooledConnection] = []
        self._lock = threading.Lock()
        self._stats = ConnectionStats()
        self._initialized = False
        self._cleanup_thread = None
        self._shutdown = False

        # Initialize pool
        self._initialize_pool()

        logger.info(f"Database connection pool initialized: size={pool_size}, overflow={max_overflow}")

    def _initialize_pool(self):
        """Create initial pool of connections"""
        with self._lock:
            # Ensure database directory exists
            db_path = Path(self.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create initial connections
            for i in range(self.pool_size):
                try:
                    conn = self._create_connection()
                    self._pool.append(PooledConnection(connection=conn))
                    self._stats.total_connections += 1
                except Exception as e:
                    logger.error(f"Failed to create connection {i}: {e}")
                    raise

            self._initialized = True

            # Start cleanup thread
            self._start_cleanup_thread()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection"""
        try:
            conn = sqlite3.Connection(
                self.database_path,
                check_same_thread=False,  # Allow use across threads
                timeout=self.timeout
            )

            # Enable Write-Ahead Logging for better concurrency
            conn.execute('PRAGMA journal_mode=WAL')

            # Set row factory for dict-like access
            conn.row_factory = sqlite3.Row

            return conn

        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise

    def _start_cleanup_thread(self):
        """Start background thread for connection cleanup"""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="ConnectionPoolCleanup"
        )
        self._cleanup_thread.start()

    def _cleanup_worker(self):
        """Background worker that cleans up old/idle connections"""
        while not self._shutdown:
            try:
                time.sleep(60)  # Check every minute

                with self._lock:
                    now = datetime.now()
                    connections_to_remove = []

                    for pooled_conn in self._pool:
                        # Skip in-use connections
                        if pooled_conn.in_use:
                            continue

                        # Remove if too old
                        if pooled_conn.age_seconds > self.max_connection_age:
                            logger.debug(f"Removing aged connection (age: {pooled_conn.age_seconds:.0f}s)")
                            connections_to_remove.append(pooled_conn)
                            continue

                        # Remove if idle too long
                        if pooled_conn.idle_seconds > self.max_idle_time:
                            logger.debug(f"Removing idle connection (idle: {pooled_conn.idle_seconds:.0f}s)")
                            connections_to_remove.append(pooled_conn)
                            continue

                    # Close and remove old connections
                    for pooled_conn in connections_to_remove:
                        try:
                            pooled_conn.connection.close()
                            self._pool.remove(pooled_conn)
                            self._stats.total_connections -= 1
                        except Exception as e:
                            logger.error(f"Error closing connection: {e}")

                    # Ensure minimum pool size
                    current_size = len(self._pool)
                    if current_size < self.pool_size:
                        for _ in range(self.pool_size - current_size):
                            try:
                                conn = self._create_connection()
                                self._pool.append(PooledConnection(connection=conn))
                                self._stats.total_connections += 1
                            except Exception as e:
                                logger.error(f"Error creating replacement connection: {e}")

            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.

        Usage:
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        start_time = time.time()
        pooled_conn = None

        try:
            # Get connection from pool
            pooled_conn = self._acquire_connection()

            wait_time = time.time() - start_time
            self._update_stats(wait_time)

            # Update connection metadata
            pooled_conn.last_used = datetime.now()
            pooled_conn.use_count += 1
            pooled_conn.thread_id = threading.get_ident()

            yield pooled_conn.connection

        except Exception as e:
            self._stats.failed_requests += 1
            logger.error(f"Error using connection: {e}")
            raise

        finally:
            # Return connection to pool
            if pooled_conn:
                self._release_connection(pooled_conn)

    def _acquire_connection(self) -> PooledConnection:
        """Acquire a connection from the pool"""
        deadline = time.time() + self.timeout

        while time.time() < deadline:
            with self._lock:
                # Find available connection
                for pooled_conn in self._pool:
                    if not pooled_conn.in_use:
                        pooled_conn.in_use = True
                        self._stats.active_connections += 1
                        self._stats.idle_connections -= 1
                        return pooled_conn

                # Can we create overflow connection?
                current_size = len(self._pool)
                max_size = self.pool_size + self.max_overflow

                if current_size < max_size:
                    try:
                        conn = self._create_connection()
                        pooled_conn = PooledConnection(connection=conn, in_use=True)
                        self._pool.append(pooled_conn)
                        self._stats.total_connections += 1
                        self._stats.active_connections += 1
                        return pooled_conn
                    except Exception as e:
                        logger.error(f"Failed to create overflow connection: {e}")

            # Wait a bit before retrying
            time.sleep(0.1)

        # Timeout - pool exhausted
        self._stats.pool_exhaustions += 1
        raise ConnectionPoolExhaustedError(pool_size=len(self._pool))

    def _release_connection(self, pooled_conn: PooledConnection):
        """Release connection back to pool"""
        with self._lock:
            pooled_conn.in_use = False
            pooled_conn.thread_id = None
            self._stats.active_connections -= 1
            self._stats.idle_connections += 1

    def _update_stats(self, wait_time: float):
        """Update pool statistics"""
        with self._lock:
            self._stats.total_requests += 1

            # Update wait time stats
            if wait_time > self._stats.max_wait_time:
                self._stats.max_wait_time = wait_time

            # Calculate moving average
            n = self._stats.total_requests
            self._stats.avg_wait_time = (
                (self._stats.avg_wait_time * (n - 1) + wait_time) / n
            )

    def get_stats(self) -> ConnectionStats:
        """Get current pool statistics"""
        with self._lock:
            stats = ConnectionStats(
                total_connections=len(self._pool),
                active_connections=sum(1 for c in self._pool if c.in_use),
                idle_connections=sum(1 for c in self._pool if not c.in_use),
                total_requests=self._stats.total_requests,
                failed_requests=self._stats.failed_requests,
                avg_wait_time=self._stats.avg_wait_time,
                max_wait_time=self._stats.max_wait_time,
                pool_exhaustions=self._stats.pool_exhaustions
            )
            return stats

    def shutdown(self):
        """Shutdown the pool and close all connections"""
        logger.info("Shutting down connection pool...")
        self._shutdown = True

        with self._lock:
            for pooled_conn in self._pool:
                try:
                    pooled_conn.connection.close()
                except Exception as e:
                    logger.error(f"Error closing connection during shutdown: {e}")

            self._pool.clear()
            self._stats.total_connections = 0
            self._stats.active_connections = 0
            self._stats.idle_connections = 0

        logger.info("Connection pool shutdown complete")


class APIKeyManager:
    """
    Validates and manages API keys.

    Features:
    - Validates keys exist before use
    - Checks key format
    - Provides helpful error messages
    - Caches validation results
    """

    REQUIRED_KEYS = {
        'ALPACA_API_KEY': {
            'description': 'Alpaca trading API key',
            'format': r'^[A-Z0-9]{20}$',
            'required_for': ['trading', 'market_data']
        },
        'ALPACA_SECRET_KEY': {
            'description': 'Alpaca trading secret key',
            'format': r'^[A-Za-z0-9]{40}$',
            'required_for': ['trading', 'market_data']
        },
        'NEWS_API_KEY': {
            'description': 'NewsAPI key for sentiment analysis',
            'format': r'^[a-f0-9]{32}$',
            'required_for': ['news', 'sentiment'],
            'optional': True
        },
        'ALPHA_VANTAGE_KEY': {
            'description': 'Alpha Vantage API key',
            'format': r'^[A-Z0-9]{16}$',
            'required_for': ['fundamentals'],
            'optional': True
        }
    }

    def __init__(self, env_file: Optional[str] = '.env'):
        """Initialize API key manager"""
        self.env_file = env_file
        self._validated_keys = {}
        self._load_env()

    def _load_env(self):
        """Load environment variables from .env file"""
        if self.env_file and Path(self.env_file).exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(self.env_file)
                logger.debug(f"Loaded environment from {self.env_file}")
            except ImportError:
                logger.warning("python-dotenv not installed, using system environment only")
            except Exception as e:
                logger.error(f"Error loading .env file: {e}")

    def validate_key(self, key_name: str, required: bool = True) -> bool:
        """
        Validate a single API key.

        Args:
            key_name: Name of environment variable
            required: Whether key is required

        Returns:
            True if valid, raises exception if invalid and required

        Raises:
            APIAuthenticationError: If key is invalid or missing (when required)
        """
        # Check cache
        if key_name in self._validated_keys:
            return self._validated_keys[key_name]

        # Get key configuration
        key_config = self.REQUIRED_KEYS.get(key_name, {})
        is_optional = key_config.get('optional', False)

        # Get key value
        key_value = os.getenv(key_name)

        # Check if missing
        if not key_value:
            if required and not is_optional:
                raise APIAuthenticationError(
                    service=key_config.get('description', key_name),
                    details={
                        'key_name': key_name,
                        'env_file': self.env_file,
                        'required_for': key_config.get('required_for', [])
                    },
                    recovery_suggestion=f"Set {key_name} in {self.env_file} file"
                )
            else:
                logger.info(f"Optional API key {key_name} not configured")
                self._validated_keys[key_name] = False
                return False

        # Check format if specified
        key_format = key_config.get('format')
        if key_format:
            import re
            if not re.match(key_format, key_value):
                raise APIAuthenticationError(
                    service=key_config.get('description', key_name),
                    details={
                        'key_name': key_name,
                        'reason': 'Invalid key format'
                    },
                    recovery_suggestion=f"Check {key_name} format in {self.env_file}"
                )

        # Valid!
        self._validated_keys[key_name] = True
        logger.debug(f"API key validated: {key_name}")
        return True

    def validate_all_required(self) -> Dict[str, bool]:
        """
        Validate all required API keys.

        Returns:
            Dictionary of key_name: valid status

        Raises:
            APIAuthenticationError: If any required key is invalid
        """
        results = {}

        for key_name, key_config in self.REQUIRED_KEYS.items():
            is_optional = key_config.get('optional', False)
            try:
                results[key_name] = self.validate_key(key_name, required=not is_optional)
            except APIAuthenticationError:
                if not is_optional:
                    raise
                results[key_name] = False

        return results

    def get_key(self, key_name: str) -> Optional[str]:
        """
        Get API key value (only if validated).

        Args:
            key_name: Name of environment variable

        Returns:
            Key value or None if not available
        """
        if self.validate_key(key_name, required=False):
            return os.getenv(key_name)
        return None

    def check_features_available(self, feature: str) -> bool:
        """
        Check if API keys for a feature are available.

        Args:
            feature: Feature name (e.g., 'trading', 'news', 'sentiment')

        Returns:
            True if all required keys for feature are available
        """
        required_keys = [
            key_name
            for key_name, config in self.REQUIRED_KEYS.items()
            if feature in config.get('required_for', [])
        ]

        for key_name in required_keys:
            if not self.validate_key(key_name, required=False):
                return False

        return True

    def get_status_report(self) -> str:
        """Get human-readable status report of API keys"""
        lines = ["API Key Status Report:", "=" * 80]

        for key_name, key_config in self.REQUIRED_KEYS.items():
            is_optional = key_config.get('optional', False)
            description = key_config.get('description', key_name)

            try:
                is_valid = self.validate_key(key_name, required=False)
                status = "✓ CONFIGURED" if is_valid else "✗ MISSING"
                required_text = "(optional)" if is_optional else "(required)"
            except Exception:
                status = "✗ INVALID"
                required_text = "(required)"

            lines.append(f"{status:20} {description:40} {required_text}")

        lines.append("=" * 80)
        return "\n".join(lines)


# Global instances
_db_pool: Optional[DatabaseConnectionPool] = None
_api_keys: Optional[APIKeyManager] = None


def get_db_pool(database_path: str = 'database/trading.db', **kwargs) -> DatabaseConnectionPool:
    """Get or create global database connection pool"""
    global _db_pool

    if _db_pool is None:
        _db_pool = DatabaseConnectionPool(database_path, **kwargs)

    return _db_pool


def get_api_manager(env_file: str = '.env') -> APIKeyManager:
    """Get or create global API key manager"""
    global _api_keys

    if _api_keys is None:
        _api_keys = APIKeyManager(env_file)

    return _api_keys


if __name__ == '__main__':
    # Test connection pool
    print("Testing Database Connection Pool...")
    print("=" * 80)

    pool = DatabaseConnectionPool('test.db', pool_size=3, max_overflow=2)

    print(f"Initial pool stats: {pool.get_stats()}")

    # Test getting connections
    with pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        print(f"✓ Connection 1 works")

    with pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 2")
        print(f"✓ Connection 2 works")

    stats = pool.get_stats()
    print(f"\nFinal stats:")
    print(f"  Total connections: {stats.total_connections}")
    print(f"  Active: {stats.active_connections}")
    print(f"  Idle: {stats.idle_connections}")
    print(f"  Total requests: {stats.total_requests}")

    pool.shutdown()

    print("\n" + "=" * 80)
    print("Testing API Key Manager...")
    print("=" * 80)

    api_mgr = APIKeyManager(env_file='.env.example')
    print(api_mgr.get_status_report())

    print("\n" + "=" * 80)
    print("Resource manager tests complete!")
