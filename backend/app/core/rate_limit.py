"""
PackVision AI - In-Memory Rate Limiting
=======================================
Small, dependency-free sliding-window rate limiter for FastAPI.

Design notes:
- NO external dependencies (no slowapi / redis). Pure stdlib.
- Background-free: expired hits are pruned lazily "on access" whenever a
  key is checked, plus an opportunistic sweep of stale keys to bound memory.
- Only the remote client IP is kept in memory (never persisted, no PII
  beyond the IP itself). State is process-local and resets on restart,
  which is acceptable for a single-process deployment.
- Limits are configurable through environment variables:
    RATE_LIMIT_UPLOADS_PER_MIN      (default 10)
    RATE_LIMIT_GENERATIONS_PER_MIN  (default 10)
    RATE_LIMIT_READ_PER_MIN         (default 120)
"""

import math
import os
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Tuple

from fastapi import HTTPException, Request, status

# Maximum distinct keys tracked before a stale-key sweep is forced.
_MAX_TRACKED_KEYS = 10_000

_DEFAULT_RETRY_AFTER = 1


def _env_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to the default on bad input."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


class RateLimiter:
    """In-memory sliding-window rate limiter keyed by an arbitrary string.

    Args:
        max_requests: Number of requests allowed per window.
        window_seconds: Length of the sliding window in seconds.
        per_key: Semantic description of the key (currently only "ip").
        enabled: When False, :meth:`check` always allows the request. Tests
            flip this off globally via a conftest fixture.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        per_key: str = "ip",
        enabled: bool = True,
    ) -> None:
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = float(window_seconds)
        self.per_key = per_key
        self.enabled = enabled
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        # Injectable clock (tests may monkeypatch this instance attribute).
        self._clock = time.monotonic

    # ------------------------------------------------------------------ #
    # Core sliding-window logic (prune on access, no background thread)
    # ------------------------------------------------------------------ #
    def check(self, key: str) -> Tuple[bool, float]:
        """Record a hit for ``key`` and decide whether it is allowed.

        Returns:
            (allowed, retry_after_seconds) — retry_after is 0.0 when allowed.
        """
        if not self.enabled:
            return True, 0.0

        now = self._clock()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = self._hits[key]
            # Lazy pruning: drop timestamps that slid out of the window.
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.max_requests:
                retry_after = max(0.0, self.window_seconds - (now - hits[0]))
                return False, retry_after

            hits.append(now)

            # Opportunistic memory bound: sweep fully-stale keys occasionally.
            if len(self._hits) > _MAX_TRACKED_KEYS:
                stale = [k for k, d in self._hits.items() if not d or d[-1] <= cutoff]
                for k in stale:
                    del self._hits[k]
                if key in stale and key not in self._hits:
                    self._hits[key] = hits

            if not hits:
                # Should not happen (we just appended), defensive cleanup.
                self._hits.pop(key, None)
            return True, 0.0

    def reset(self) -> None:
        """Clear all tracked hit windows (used by tests)."""
        with self._lock:
            self._hits.clear()


def client_key(request: Request, per_key: str = "ip") -> str:
    """Derive the limiter bucket key from the request.

    Only the remote IP is used; no headers that could carry PII beyond the
    connection peer are consulted. ``"unknown"`` is used when the ASGI server
    does not populate ``request.client`` (e.g. some test transports).
    """
    if per_key != "ip":  # Future-proofing: only "ip" is supported today.
        per_key = "ip"
    host = request.client.host if request.client else None
    return host or "unknown"


def rate_limit_dependency(limiter: RateLimiter) -> Callable:
    """Build a FastAPI dependency enforcing the given limiter.

    Usage on a route: ``deps = Depends(rate_limit_dependency(read_limiter))``.
    Raises 429 with a ``Retry-After`` header when the bucket is exhausted.
    """

    async def _enforce_rate_limit(request: Request) -> None:
        if not limiter.enabled:
            return
        allowed, retry_after = limiter.check(client_key(request, limiter.per_key))
        if not allowed:
            seconds = max(_DEFAULT_RETRY_AFTER, int(math.ceil(retry_after)))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    "Rate limit exceeded. "
                    f"Please retry in {seconds} second(s)."
                ),
                headers={"Retry-After": str(seconds)},
            )

    return _enforce_rate_limit


def rate_limit_response_body(retry_after: float) -> Tuple[str, int]:
    """Return (message, seconds) for middleware-level 429 responses."""
    seconds = max(_DEFAULT_RETRY_AFTER, int(math.ceil(retry_after)))
    return (
        f"Rate limit exceeded. Please retry in {seconds} second(s).",
        seconds,
    )


# ---------------------------------------------------------------------- #
# Shared limiters (env-overridable, always enabled — including production)
# ---------------------------------------------------------------------- #
_UPLOADS_PER_MIN = _env_int("RATE_LIMIT_UPLOADS_PER_MIN", 10)
_GENERATIONS_PER_MIN = _env_int("RATE_LIMIT_GENERATIONS_PER_MIN", 10)
_READS_PER_MIN = _env_int("RATE_LIMIT_READ_PER_MIN", 120)

upload_limiter = RateLimiter(_UPLOADS_PER_MIN, 60.0, per_key="ip")
generation_create_limiter = RateLimiter(_GENERATIONS_PER_MIN, 60.0, per_key="ip")
read_limiter = RateLimiter(_READS_PER_MIN, 60.0, per_key="ip")

# Pre-built FastAPI dependencies for the API endpoints owned by this module set.
rate_limit_generation_create = rate_limit_dependency(generation_create_limiter)
rate_limit_read = rate_limit_dependency(read_limiter)
