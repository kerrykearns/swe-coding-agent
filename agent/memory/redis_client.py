"""Connecting to Redis for trajectory memory (M7).

Configuration is read from the environment (``REDIS_HOST`` / ``REDIS_PORT``,
``.env`` loaded once at import time), exactly as
:mod:`agent.core.llm_client` and :mod:`agent.core.github_client` do for their
own backends. Nothing here is validated until :func:`get_redis_client` is
actually called, so importing this module can never fail without Redis
running — the test suite imports it to unit-test the rest of ``agent.memory``
against an injected :class:`fakeredis.FakeRedis` instead.
"""

from __future__ import annotations

import os
from typing import Optional

import redis
from dotenv import load_dotenv

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "get_redis_client"]

load_dotenv()

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 6379


def get_redis_client(client: Optional[redis.Redis] = None) -> redis.Redis:
    """Return a Redis client, reading ``REDIS_HOST``/``REDIS_PORT`` if not given.

    Args:
        client: An already-built client to use instead of connecting for real —
            typically a :class:`fakeredis.FakeRedis` in tests, or a real
            ``redis.Redis`` a caller already holds. Passed straight through
            unchanged.

    Returns:
        A ``redis.Redis`` client with ``decode_responses=True``, so values read
        back from Redis are ``str``, not ``bytes`` — everything stored by
        :mod:`agent.memory.trajectory_store` is JSON text. Connecting lazily:
        no network call is made here, only when a command is first issued.
    """
    if client is not None:
        return client

    host = os.getenv("REDIS_HOST") or DEFAULT_HOST
    port = int(os.getenv("REDIS_PORT") or DEFAULT_PORT)
    return redis.Redis(host=host, port=port, decode_responses=True)
