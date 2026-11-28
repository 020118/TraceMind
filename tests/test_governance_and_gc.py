from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from agent.bus.events import OutboundMessage
from agent.reliability import ReliabilityStore
from api.app import create_app
from api.governance import (
    SlidingWindowRateLimiter,
    cleanup_expired_attachments,
    evaluate_daily_budget,
)


def test_sliding_window_rate_limiter():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=10.0)
    assert limiter.check("client-a", now=100.0) == (True, 0)
    assert limiter.check("client-a", now=101.0) == (True, 0)
    allowed, retry_after = limiter.check("client-a", now=102.0)
    assert allowed is False
    assert retry_after >= 1
    # Different client is unaffected
    assert limiter.check("client-b", now=102.0) == (True, 0)
    # After window passes, client-a is allowed again
    assert limiter.check("client-a", now=111.0) == (True, 0)


def test_cleanup_expired_attachments(tmp_path: Path):
    base = tmp_path / "chat_attachments"
    old_dir = base / "session_old"
    new_dir = base / "session_new"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)

    old_file = old_dir / "expired.png"
    new_file = new_dir / "fresh.png"
    old_file.write_bytes(b"old")
    new_file.write_bytes(b"new")

    now = time.time()
    ten_days_ago = now - 10 * 86400
    os.utime(old_file, (ten_days_ago, ten_days_ago))
    os.utime(new_file, (now, now))

    removed = cleanup_expired_attachments(tmp_path, ttl_days=7, now=now)
    assert removed == 1
    assert not old_file.exists()
    assert not old_dir.exists()
    assert new_file.exists()


def test_today_cost_and_budget_evaluation(tmp_path: Path):
    store = ReliabilityStore(tmp_path / "reliability.db")
    try:
        assert store.today_cost_usd() == 0.0
        store.record_model_usage(
            trace_id="t1",
            session_key="api:s1",
            model="test-model",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            estimated_cost_usd=0.25,
            purpose="chat",
        )
        store.record_model_usage(
            trace_id="t2",
            session_key="api:s1",
            model="test-model",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            estimated_cost_usd=0.30,
            purpose="chat",
        )
        assert abs(store.today_cost_usd() - 0.55) < 1e-6

        # Below budget -> keep requested mode
        mode, blocked, today, limit = evaluate_daily_budget(
            store, requested_mode="deep", budget_usd=1.0, action="downgrade"
        )
        assert (mode, blocked, limit) == ("deep", False, 1.0)
        assert abs(today - 0.55) < 1e-6

        # Exceeded + downgrade -> fast
        mode, blocked, _, _ = evaluate_daily_budget(
            store, requested_mode="deep", budget_usd=0.5, action="downgrade"
        )
        assert (mode, blocked) == ("fast", False)

        # Exceeded + block -> blocked
        mode, blocked, _, _ = evaluate_daily_budget(
            store, requested_mode="deep", budget_usd=0.5, action="block"
        )
        assert blocked is True
    finally:
        store.close()


class _RecordingApiChannel:
    def __init__(self) -> None:
        self.modes: list[str] = []

    async def request(
        self,
        *,
        session_id: str,
        content: str,
        sender_id: str | None = None,
        timeout: float = 30.0,
        mode: str = "balanced",
        **_kwargs: Any,
    ) -> tuple[str, OutboundMessage]:
        self.modes.append(mode)
        return (
            "req-1",
            OutboundMessage(channel="api", chat_id=session_id, content=f"mode:{mode}"),
        )


class _DummyRuntime:
    def __init__(self, store: ReliabilityStore, workspace: Path) -> None:
        self.ready = True
        self.reliability_store = store
        self.workspace = workspace
        self.api_channel = _RecordingApiChannel()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def status(self) -> dict[str, Any]:
        return {"ready": True, "llm": True, "memory": False, "proactive": False}


def test_api_rate_limit_and_budget_guard(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.setenv("AGENT_RATE_LIMIT_RPM", "2")
    monkeypatch.setenv("AGENT_DAILY_BUDGET_USD", "0.10")
    monkeypatch.setenv("AGENT_BUDGET_ACTION", "downgrade")

    store = ReliabilityStore(tmp_path / "reliability.db")
    store.record_model_usage(
        trace_id="t-api",
        session_key="api:s1",
        model="m",
        prompt_tokens=100,
        completion_tokens=100,
        total_tokens=200,
        estimated_cost_usd=0.20,
        purpose="chat",
    )
    runtime = _DummyRuntime(store, tmp_path)
    app = create_app(runtime_factory=lambda: runtime)

    try:
        with TestClient(app) as client:
            r1 = client.post(
                "/v1/chat",
                headers={"X-API-Key": "k"},
                json={"session_id": "s1", "message": "hi", "mode": "deep"},
            )
            assert r1.status_code == 200
            # Downgraded from deep -> fast due to daily budget
            assert r1.json()["answer"] == "mode:fast"

            r2 = client.post(
                "/v1/chat",
                headers={"X-API-Key": "k"},
                json={"session_id": "s1", "message": "hi2"},
            )
            assert r2.status_code == 200

            # 3rd request within 60s hits rate limit 429
            r3 = client.post(
                "/v1/chat",
                headers={"X-API-Key": "k"},
                json={"session_id": "s1", "message": "hi3"},
            )
            assert r3.status_code == 429
            assert "Retry-After" in r3.headers

            # /v1/usage exposes today_cost_usd & daily_budget_usd
            usage = client.get("/v1/usage", headers={"X-API-Key": "k"})
            assert usage.status_code == 200
            assert usage.json()["today_cost_usd"] >= 0.20
            assert usage.json()["daily_budget_usd"] == 0.10
    finally:
        store.close()
