from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from shared.config import get_settings
from shared.logging import get_logger

log = get_logger(__name__)


class RateLimitedError(Exception):
    """Upstream told us to back off. Retryable."""


@dataclass
class _TokenBucket:
    """Simple async token bucket.

    EDGAR and Mapillary both publish hard request ceilings and both will
    blackhole a client that ignores them, so throttling is a correctness
    concern here rather than politeness.
    """

    rate_per_sec: float
    capacity: float = 1.0

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._updated) * self.rate_per_sec
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self._tokens) / self.rate_per_sec)


class CachedClient:
    """Rate-limited HTTP client with an on-disk response cache.

    The cache is what makes the eval suite reproducible and cheap: an ingest
    run replayed during development hits disk, not the upstream API, so a
    retrieval ablation does not depend on the SEC being up.
    """

    def __init__(
        self,
        base_url: str = "",
        *,
        headers: dict[str, str] | None = None,
        rate_per_sec: float = 5.0,
        cache_namespace: str = "http",
        timeout: float = 30.0,
        use_cache: bool = True,
    ) -> None:
        self._bucket = _TokenBucket(rate_per_sec=rate_per_sec)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers or {},
            timeout=timeout,
            follow_redirects=True,
        )
        self._use_cache = use_cache
        self._cache_dir = get_settings().resolved_cache_dir() / cache_namespace
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self) -> CachedClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _cache_path(self, method: str, url: str, params: dict[str, Any] | None) -> Path:
        key = json.dumps(
            {"m": method, "u": url, "p": params or {}}, sort_keys=True, default=str
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._cache_dir / f"{digest}.json"

    @retry(
        retry=retry_if_exception_type((RateLimitedError, httpx.TransportError)),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _fetch(
        self, method: str, url: str, params: dict[str, Any] | None
    ) -> httpx.Response:
        await self._bucket.acquire()
        response = await self._client.request(method, url, params=params)
        if response.status_code in (429, 503):
            raise RateLimitedError(f"{response.status_code} from {url}")
        response.raise_for_status()
        return response

    async def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        refresh: bool = False,
    ) -> Any:
        path = self._cache_path("GET", url, params)
        if self._use_cache and not refresh and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        response = await self._fetch("GET", url, params)
        payload = response.json()
        if self._use_cache:
            path.write_text(json.dumps(payload), encoding="utf-8")
        log.debug("http.fetch", url=url, status=response.status_code, cached=False)
        return payload

    async def get_bytes(
        self, url: str, params: dict[str, Any] | None = None, *, refresh: bool = False
    ) -> bytes:
        path = self._cache_path("GET", url, params).with_suffix(".bin")
        if self._use_cache and not refresh and path.exists():
            return path.read_bytes()

        response = await self._fetch("GET", url, params)
        if self._use_cache:
            path.write_bytes(response.content)
        return response.content
