"""API governance utilities: sliding-window rate limiting, budget guard, and attachment GC."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding-window rate limiter keyed by client."""

    def __init__(self, *, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        self.max_requests = int(max_requests)
        self.window_seconds = float(window_seconds)
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)``."""
        if self.max_requests <= 0:
            return True, 0
        ts = time.monotonic() if now is None else now
        cutoff = ts - self.window_seconds
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = deque()
                self._buckets[key] = bucket
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (ts - bucket[0])) + 1)
                return False, retry_after
            bucket.append(ts)
            # Evict stale keys periodically to bound memory.
            if len(self._buckets) > 2048:
                stale = [k for k, q in self._buckets.items() if not q or q[-1] <= cutoff]
                for k in stale:
                    del self._buckets[k]
        return True, 0


def parse_float_env(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def parse_int_env(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def evaluate_daily_budget(
    store: Any,
    *,
    requested_mode: str,
    budget_usd: float | None = None,
    action: str | None = None,
) -> tuple[str, bool, float, float]:
    """Evaluate today's LLM spend against ``AGENT_DAILY_BUDGET_USD``.

    Returns ``(effective_mode, should_block, today_cost, budget_limit)``.
    """
    limit = (
        parse_float_env("AGENT_DAILY_BUDGET_USD", 0.0)
        if budget_usd is None
        else float(budget_usd)
    )
    if limit <= 0 or store is None or not hasattr(store, "today_cost_usd"):
        return requested_mode, False, 0.0, max(0.0, limit)

    today_cost = float(store.today_cost_usd())
    if today_cost < limit:
        return requested_mode, False, today_cost, limit

    policy = (
        (action if action is not None else os.environ.get("AGENT_BUDGET_ACTION", "downgrade"))
        .strip()
        .lower()
    )
    if policy == "block":
        return requested_mode, True, today_cost, limit
    logger.warning(
        "Daily LLM budget reached ($%.4f / $%.4f); downgrading mode %s -> fast",
        today_cost,
        limit,
        requested_mode,
    )
    return "fast", False, today_cost, limit


def cleanup_expired_attachments(
    workspace: Path | None,
    *,
    ttl_days: int | None = None,
    now: float | None = None,
) -> int:
    """Delete uploaded chat attachments older than ``ttl_days``."""
    if workspace is None:
        return 0
    days = (
        parse_int_env("AGENT_ATTACHMENT_TTL_DAYS", 7)
        if ttl_days is None
        else int(ttl_days)
    )
    if days <= 0:
        return 0
    base = (Path(workspace).resolve() / "chat_attachments").resolve()
    if not base.is_dir():
        return 0

    cutoff = (time.time() if now is None else now) - (days * 86400)
    removed = 0
    for path in list(base.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue

    # Prune empty session subdirectories.
    for sub in sorted(base.rglob("*"), reverse=True):
        if sub.is_dir():
            try:
                sub.rmdir()
            except OSError:
                pass
    if removed:
        logger.info("Cleaned up %d expired chat attachment(s) older than %dd", removed, days)
    return removed
