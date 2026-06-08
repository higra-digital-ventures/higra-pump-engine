"""PostgreSQL connection pool for HPE.

Uses psycopg2 ThreadedConnectionPool.
Credentials read from environment / .env file.
"""
from __future__ import annotations

import os
import threading
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import logging

log = logging.getLogger(__name__)

_pool: ThreadedConnectionPool | None = None
_lock = threading.Lock()


def _get_params() -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "higra123"),
        "dbname": os.getenv("DB_NAME", "db_pump_engine"),
        "connect_timeout": 5,
        "sslmode": "disable",
    }


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool and not _pool.closed:
        return _pool
    with _lock:
        if _pool and not _pool.closed:
            return _pool
        params = _get_params()
        _pool = ThreadedConnectionPool(
            minconn=int(os.getenv("DB_POOL_MIN", "2")),
            maxconn=int(os.getenv("DB_POOL_MAX", "10")),
            **params,
        )
        log.info("HPE connection pool created (host=%s db=%s)", params["host"], params["dbname"])
        return _pool


class _PooledConn:
    def __init__(self, conn, pool):
        self._conn = conn
        self._pool = pool
        self._returned = False

    def close(self):
        if not self._returned:
            self._returned = True
            try:
                self._pool.putconn(self._conn)
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def get_connection():
    """Get a pooled connection. Call .close() to return it to pool."""
    try:
        pool = _get_pool()
        conn = pool.getconn()
        return _PooledConn(conn, pool)
    except Exception:
        log.warning("Pool unavailable, using direct connection")
        return psycopg2.connect(**_get_params())


def test_connection() -> bool:
    """Test DB connectivity. Returns True if OK."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                ver = cur.fetchone()
                log.info("HPE DB connected: %s", ver)
                return True
    except Exception as e:
        log.error("HPE DB connection failed: %s", e)
        return False
