"""FastAPI entry point for the TraceMind HTTP service."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent.runtime import AgentRuntime
from agent.reliability.journal import PENDING_CONFIRMATION
from api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    OperationDecisionRequest,
    PendingConfirmation,
    SessionCreateResponse,
    SessionHistoryResponse,
    SessionListResponse,
    SessionMessage,
    SessionSummary,
    ToolTraceItem,
)

RuntimeFactory = Callable[[], Any]


def _flatten_trace(metadata: dict[str, Any] | None) -> list[ToolTraceItem]:
    trace: list[ToolTraceItem] = []
    for group in (metadata or {}).get("tool_chain", []) or []:
        for call in group.get("calls", []) or []:
            trace.append(
                ToolTraceItem(
                    call_id=str(call.get("call_id", "")),
                    name=str(call.get("name", "unknown")),
                    arguments=dict(call.get("arguments") or {}),
                    result=str(call.get("result", "")),
                    status=str(call.get("status", "succeeded")),
                    duration_ms=max(0, int(call.get("duration_ms", 0) or 0)),
                    operation_id=str(call.get("operation_id", "")),
                )
            )
    return trace


def _reliability_store(runtime: Any) -> Any:
    store = getattr(runtime, "reliability_store", None)
    if store is not None:
        return store
    loop = getattr(runtime, "agent_loop", None)
    return getattr(loop, "reliability", None)


def _pending_for_session(runtime: Any, session_id: str) -> PendingConfirmation | None:
    store = _reliability_store(runtime)
    if store is None:
        return None
    record = store.get_pending_operation(f"api:{session_id}")
    if record is None:
        return None
    return PendingConfirmation(
        operation_id=record.operation_id,
        tool_name=record.tool_name,
        arguments=dict(record.arguments),
        status=record.status,
        created_at=record.created_at,
        expires_at=record.expires_at,
    )


def _chat_response(
    *, runtime: Any, request_id: str, session_id: str, outbound: Any, started: float
) -> ChatResponse:
    return ChatResponse(
        request_id=request_id,
        session_id=session_id,
        answer=outbound.content,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        trace=_flatten_trace(getattr(outbound, "metadata", None)),
        pending_confirmation=_pending_for_session(runtime, session_id),
    )


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def create_app(
    *,
    runtime_factory: RuntimeFactory | None = None,
    config_path: str | None = None,
) -> FastAPI:
    """Create an application; dependency injection keeps API tests offline."""

    resolved_config = config_path or os.environ.get("AGENT_CONFIG", "config.toml")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = (
            runtime_factory()
            if runtime_factory is not None
            else AgentRuntime.from_config_path(
                resolved_config,
                enable_console=False,
                enable_telegram=_env_flag("AGENT_ENABLE_TELEGRAM"),
                enable_proactive=_env_flag("AGENT_ENABLE_PROACTIVE"),
            )
        )
        app.state.runtime = runtime
        await runtime.start()
        try:
            yield
        finally:
            await runtime.stop()

    application = FastAPI(
        title="TraceMind API",
        description="HTTP interface for the asynchronous personal Agent runtime.",
        version="0.1.0",
        lifespan=lifespan,
    )
    cors_origins = [
        item.strip()
        for item in os.environ.get(
            "AGENT_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if item.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    async def verify_api_key(x_api_key: str = Header(default="")) -> None:
        expected = os.environ.get("AGENT_API_KEY", "").strip()
        if expected and not secrets.compare_digest(x_api_key, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get("/ready", tags=["system"])
    async def ready(request: Request) -> dict[str, Any]:
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent runtime is not initialized",
            )
        snapshot = runtime.status()
        if not snapshot.get("ready", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=snapshot,
            )
        return snapshot

    @application.get(
        "/v1/rag/metrics",
        tags=["rag"],
        dependencies=[Depends(verify_api_key)],
    )
    async def rag_metrics(request: Request, limit: int = 500) -> dict[str, Any]:
        runtime = request.app.state.runtime
        service = getattr(runtime, "rag_service", None)
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service is unavailable",
            )
        return service.retrieval_metrics(max(1, min(limit, 5000)))

    @application.post(
        "/v1/chat",
        response_model=ChatResponse,
        tags=["chat"],
        dependencies=[Depends(verify_api_key)],
    )
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        runtime = request.app.state.runtime
        if not runtime.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent runtime is not ready",
            )

        started = time.perf_counter()
        try:
            request_id, outbound = await runtime.api_channel.request(
                session_id=payload.session_id,
                sender_id=payload.sender_id,
                content=payload.message,
                timeout=payload.timeout_seconds,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Agent response timeout",
            ) from exc

        return _chat_response(
            runtime=runtime,
            request_id=request_id,
            session_id=payload.session_id,
            outbound=outbound,
            started=started,
        )

    @application.get(
        "/v1/sessions",
        response_model=SessionListResponse,
        tags=["chat"],
        dependencies=[Depends(verify_api_key)],
    )
    async def list_sessions(request: Request) -> SessionListResponse:
        manager = getattr(request.app.state.runtime, "session_manager", None)
        if manager is None:
            return SessionListResponse()
        return SessionListResponse(
            sessions=[
                SessionSummary(**item)
                for item in manager.list_dashboard_sessions(channel="api", limit=100)
            ]
        )

    @application.post(
        "/v1/sessions",
        response_model=SessionCreateResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["chat"],
        dependencies=[Depends(verify_api_key)],
    )
    async def create_session(request: Request) -> SessionCreateResponse:
        manager = getattr(request.app.state.runtime, "session_manager", None)
        if manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session store is unavailable",
            )

        # UUID4 already makes collisions vanishingly unlikely; the SQLite
        # primary key remains the authoritative uniqueness check.
        for _ in range(5):
            session_id = f"web-{uuid.uuid4()}"
            try:
                session = manager.create_session(
                    f"api:{session_id}",
                    metadata={"source": "dashboard"},
                )
            except ValueError:
                continue
            return SessionCreateResponse(
                session=SessionSummary(
                    session_id=session_id,
                    title="新会话",
                    preview="",
                    created_at=session.created_at.isoformat(),
                    updated_at=session.updated_at.isoformat(),
                    message_count=0,
                )
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to allocate a unique session id",
        )

    @application.post(
        "/v1/chat/stream",
        tags=["chat"],
        dependencies=[Depends(verify_api_key)],
    )
    async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
        """Stream observable Agent events and model answer deltas as NDJSON."""

        runtime = request.app.state.runtime
        if not runtime.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent runtime is not ready",
            )

        started = time.perf_counter()

        async def generate():
            try:
                async for request_id, event in runtime.api_channel.request_stream(
                    session_id=payload.session_id,
                    sender_id=payload.sender_id,
                    content=payload.message,
                    timeout=payload.timeout_seconds,
                ):
                    if event.get("type") == "turn_finished":
                        response = _chat_response(
                            runtime=runtime,
                            request_id=request_id,
                            session_id=payload.session_id,
                            outbound=event["outbound"],
                            started=started,
                        )
                        wire_event = {
                            "type": "turn_finished",
                            "response": response.model_dump(),
                        }
                    else:
                        wire_event = event
                    yield json.dumps(wire_event, ensure_ascii=False) + "\n"
            except (TimeoutError, asyncio.TimeoutError):
                yield json.dumps(
                    {"type": "error", "message": "Agent response timeout"},
                    ensure_ascii=False,
                ) + "\n"

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    async def decide_operation(
        operation_id: str,
        payload: OperationDecisionRequest,
        request: Request,
        *,
        decision: str,
    ) -> ChatResponse:
        runtime = request.app.state.runtime
        store = _reliability_store(runtime)
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Operation journal is unavailable",
            )
        operation = store.get_operation(operation_id)
        expected_session = f"api:{payload.session_id}"
        if operation is None or operation.session_key != expected_session:
            raise HTTPException(status_code=404, detail="Operation not found")
        if operation.status != PENDING_CONFIRMATION:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Operation is already {operation.status}",
            )
        pending = store.get_pending_operation(expected_session)
        if pending is None or pending.operation_id != operation_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Operation is no longer the active confirmation",
            )

        started = time.perf_counter()
        content = "确认删除" if decision == "confirm" else "取消删除"
        try:
            request_id, outbound = await runtime.api_channel.request(
                session_id=payload.session_id,
                sender_id="dashboard",
                content=content,
                timeout=payload.timeout_seconds,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Agent response timeout",
            ) from exc
        return _chat_response(
            runtime=runtime,
            request_id=request_id,
            session_id=payload.session_id,
            outbound=outbound,
            started=started,
        )

    @application.post(
        "/v1/operations/{operation_id}/confirm",
        response_model=ChatResponse,
        tags=["operations"],
        dependencies=[Depends(verify_api_key)],
    )
    async def confirm_operation(
        operation_id: str,
        payload: OperationDecisionRequest,
        request: Request,
    ) -> ChatResponse:
        return await decide_operation(
            operation_id, payload, request, decision="confirm"
        )

    @application.post(
        "/v1/operations/{operation_id}/cancel",
        response_model=ChatResponse,
        tags=["operations"],
        dependencies=[Depends(verify_api_key)],
    )
    async def cancel_operation(
        operation_id: str,
        payload: OperationDecisionRequest,
        request: Request,
    ) -> ChatResponse:
        return await decide_operation(
            operation_id, payload, request, decision="cancel"
        )

    @application.get(
        "/v1/sessions/{session_id}/messages",
        response_model=SessionHistoryResponse,
        tags=["chat"],
        dependencies=[Depends(verify_api_key)],
    )
    async def session_messages(
        session_id: str,
        request: Request,
    ) -> SessionHistoryResponse:
        manager = getattr(request.app.state.runtime, "session_manager", None)
        if manager is None:
            return SessionHistoryResponse(
                session_id=session_id,
                pending_confirmation=_pending_for_session(
                    request.app.state.runtime, session_id
                ),
            )
        session = manager.get_existing(f"api:{session_id}")
        if session is None:
            return SessionHistoryResponse(
                session_id=session_id,
                pending_confirmation=_pending_for_session(
                    request.app.state.runtime, session_id
                ),
            )
        messages = []
        for item in session.messages:
            role = str(item.get("role", ""))
            if role not in {"user", "assistant"}:
                continue
            messages.append(
                SessionMessage(
                    role=role,
                    content=str(item.get("content", "")),
                    timestamp=str(item.get("timestamp", "")),
                    trace=_flatten_trace(
                        {"tool_chain": item.get("tool_chain", [])}
                    ),
                )
            )
        return SessionHistoryResponse(
            session_id=session_id,
            messages=messages,
            pending_confirmation=_pending_for_session(
                request.app.state.runtime, session_id
            ),
        )

    frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    assets_dir = frontend_dist / "assets"
    if assets_dir.is_dir():
        application.mount(
            "/assets", StaticFiles(directory=assets_dir), name="dashboard-assets"
        )

    @application.get("/", include_in_schema=False)
    async def dashboard() -> Any:
        index_file = frontend_dist / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return {
            "service": "TraceMind API",
            "dashboard": "Run the frontend dev server or build frontend/dist",
            "docs": "/docs",
        }

    return application


app = create_app()
