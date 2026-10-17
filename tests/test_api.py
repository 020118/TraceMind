from __future__ import annotations

import asyncio
import json

from fastapi.testclient import TestClient

from agent.bus.events import OutboundMessage
from agent.reliability import ReliabilityStore
from agent.reliability.journal import (
    CANCELLED,
    CONFIRMED,
    PENDING_CONFIRMATION,
    RUNNING,
    SUCCEEDED,
)
from api.app import _env_flag, create_app
from session.manager import SessionManager


class _FakeApiChannel:
    async def request(
        self,
        *,
        session_id: str,
        content: str,
        sender_id: str | None,
        timeout: float,
    ) -> tuple[str, OutboundMessage]:
        return (
            "request-123",
            OutboundMessage(
                channel="api",
                chat_id="request-123",
                content=f"echo:{content}",
            ),
        )

    async def request_stream(self, **kwargs):
        request_id, outbound = await self.request(**kwargs)
        yield request_id, {"type": "turn_started"}
        yield request_id, {"type": "answer_delta", "delta": outbound.content}
        yield request_id, {"type": "turn_finished", "outbound": outbound}


class _TimeoutApiChannel:
    async def request(self, **kwargs):
        raise asyncio.TimeoutError


class _FakeRuntime:
    def __init__(self, *, ready: bool = True, timeout: bool = False) -> None:
        self.ready = ready
        self.api_channel = _TimeoutApiChannel() if timeout else _FakeApiChannel()
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "llm": True,
            "memory": False,
            "proactive": False,
        }


class _DecisionApiChannel:
    def __init__(self, store: ReliabilityStore) -> None:
        self.store = store

    async def request(
        self,
        *,
        session_id: str,
        content: str,
        sender_id: str | None,
        timeout: float,
    ) -> tuple[str, OutboundMessage]:
        pending = self.store.get_pending_operation(f"api:{session_id}")
        assert pending is not None
        if content == "取消删除":
            self.store.transition_operation(
                pending.operation_id,
                to_status=CANCELLED,
                allowed_from={PENDING_CONFIRMATION},
                result="用户取消删除",
            )
            result = "已取消删除操作，文件没有被修改。"
            result_status = "cancelled"
        else:
            self.store.transition_operation(
                pending.operation_id,
                to_status=CONFIRMED,
                allowed_from={PENDING_CONFIRMATION},
            )
            self.store.transition_operation(
                pending.operation_id,
                to_status=RUNNING,
                allowed_from={CONFIRMED},
            )
            self.store.finish_operation(
                pending.operation_id,
                status=SUCCEEDED,
                result="已删除文件",
            )
            result = "已删除文件"
            result_status = "succeeded"
        return (
            "decision-123",
            OutboundMessage(
                channel="api",
                chat_id="decision-123",
                content=result,
                metadata={
                    "tool_chain": [
                        {
                            "calls": [
                                {
                                    "call_id": pending.operation_id,
                                    "name": pending.tool_name,
                                    "arguments": pending.arguments,
                                    "result": result,
                                    "status": result_status,
                                    "duration_ms": 2,
                                    "operation_id": pending.operation_id,
                                }
                            ]
                        }
                    ]
                },
            ),
        )


class _OperationRuntime(_FakeRuntime):
    def __init__(self, store: ReliabilityStore) -> None:
        super().__init__()
        self.reliability_store = store
        self.api_channel = _DecisionApiChannel(store)

    async def stop(self) -> None:
        self.stopped = True


class _SessionRuntime(_FakeRuntime):
    def __init__(self, workspace) -> None:
        super().__init__()
        self.session_manager = SessionManager(workspace)

    async def stop(self) -> None:
        self.session_manager.close()
        self.stopped = True


def test_env_flag_defaults_off_and_accepts_true(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_ENABLE_TELEGRAM", raising=False)
    assert _env_flag("AGENT_ENABLE_TELEGRAM") is False

    monkeypatch.setenv("AGENT_ENABLE_TELEGRAM", "true")
    assert _env_flag("AGENT_ENABLE_TELEGRAM") is True


def test_health_and_lifespan() -> None:
    runtime = _FakeRuntime()
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        assert runtime.started is True
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    assert runtime.stopped is True


def test_ready_returns_runtime_status() -> None:
    runtime = _FakeRuntime()
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["llm"] is True


def test_chat_success() -> None:
    app = create_app(runtime_factory=_FakeRuntime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"session_id": "user-1", "message": "hello"},
        )

    assert response.status_code == 200
    assert response.json()["request_id"] == "request-123"
    assert response.json()["session_id"] == "user-1"
    assert response.json()["answer"] == "echo:hello"
    assert response.json()["elapsed_ms"] >= 0
    assert response.json()["trace"] == []
    assert response.json()["pending_confirmation"] is None


def test_chat_stream_returns_ndjson_events_and_final_response() -> None:
    app = create_app(runtime_factory=_FakeRuntime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/stream",
            json={"session_id": "user-1", "message": "hello"},
        )

    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert [event["type"] for event in events] == [
        "turn_started",
        "answer_delta",
        "turn_finished",
    ]
    assert events[-1]["response"]["answer"] == "echo:hello"


def test_session_create_list_order_and_isolated_history(tmp_path) -> None:
    runtime = _SessionRuntime(tmp_path)
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        first = client.post("/v1/sessions")
        second = client.post("/v1/sessions")
        first_id = first.json()["session"]["session_id"]
        second_id = second.json()["session"]["session_id"]

        assert first.status_code == 201
        assert second.status_code == 201
        assert first_id != second_id

        first_session = runtime.session_manager.get_existing(f"api:{first_id}")
        assert first_session is not None
        first_session.add_message("user", "讨论 Agent 会话隔离")
        first_session.add_message("assistant", "每个会话保存独立历史。")
        runtime.session_manager.save(first_session)

        listing = client.get("/v1/sessions")
        first_history = client.get(f"/v1/sessions/{first_id}/messages")
        second_history = client.get(f"/v1/sessions/{second_id}/messages")

    sessions = listing.json()["sessions"]
    assert sessions[0]["session_id"] == first_id
    assert sessions[0]["title"] == "讨论 Agent 会话隔离"
    assert sessions[0]["message_count"] == 2
    assert [item["content"] for item in first_history.json()["messages"]] == [
        "讨论 Agent 会话隔离",
        "每个会话保存独立历史。",
    ]
    assert second_history.json()["messages"] == []


def test_reading_unknown_session_does_not_create_empty_row(tmp_path) -> None:
    runtime = _SessionRuntime(tmp_path)
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.get("/v1/sessions/not-created/messages")
        listing = client.get("/v1/sessions")

    assert response.status_code == 200
    assert response.json()["messages"] == []
    assert listing.json()["sessions"] == []


def test_operation_confirmation_is_scoped_and_returns_trace(tmp_path) -> None:
    store = ReliabilityStore(tmp_path / "reliability.db")
    pending = store.create_pending_operation(
        session_key="api:user-1",
        channel="api",
        chat_id="request-1",
        tool_name="delete_file",
        arguments={"path": "remove_me.txt"},
    )
    runtime = _OperationRuntime(store)
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        wrong_session = client.post(
            f"/v1/operations/{pending.operation_id}/confirm",
            json={"session_id": "other-user"},
        )
        response = client.post(
            f"/v1/operations/{pending.operation_id}/confirm",
            json={"session_id": "user-1"},
        )

    assert wrong_session.status_code == 404
    assert response.status_code == 200
    assert response.json()["pending_confirmation"] is None
    assert response.json()["trace"][0]["name"] == "delete_file"
    assert response.json()["trace"][0]["operation_id"] == pending.operation_id
    assert store.get_operation(pending.operation_id).status == SUCCEEDED
    store.close()


def test_chat_validates_empty_message() -> None:
    app = create_app(runtime_factory=_FakeRuntime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"session_id": "user-1", "message": ""},
        )

    assert response.status_code == 422


def test_chat_requires_configured_api_key(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "secret")
    app = create_app(runtime_factory=_FakeRuntime)

    with TestClient(app) as client:
        unauthorized = client.post(
            "/v1/chat",
            json={"session_id": "user-1", "message": "hello"},
        )
        authorized = client.post(
            "/v1/chat",
            headers={"X-API-Key": "secret"},
            json={"session_id": "user-1", "message": "hello"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_chat_timeout_returns_504() -> None:
    runtime = _FakeRuntime(timeout=True)
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"session_id": "user-1", "message": "hello"},
        )

    assert response.status_code == 504


def test_unready_runtime_returns_503() -> None:
    runtime = _FakeRuntime(ready=False)
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        ready_response = client.get("/ready")
        chat_response = client.post(
            "/v1/chat",
            json={"session_id": "user-1", "message": "hello"},
        )

    assert ready_response.status_code == 503
    assert chat_response.status_code == 503
