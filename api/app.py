"""FastAPI entry point for the TraceMind HTTP service."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import secrets
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent.runtime import AgentRuntime
from agent.reliability.journal import (
    PENDING_CONFIRMATION,
    TASK_CANCELLED,
    TASK_FAILED,
    TASK_INTERRUPTED,
    TASK_NEEDS_REVIEW,
)
from api.models import (
    ChatAttachment,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    OperationDecisionRequest,
    PendingConfirmation,
    SessionCreateResponse,
    SessionDeleteResponse,
    SessionHistoryResponse,
    SessionListResponse,
    SessionMessage,
    SessionSummary,
    SessionTruncateRequest,
    SessionTruncateResponse,
    SessionUpdateRequest,
    TaskRunDetailResponse,
    TaskRunListResponse,
    TaskRunSummary,
    TaskResumeResponse,
    ToolTraceItem,
    KnowledgeDocument,
    KnowledgeDocumentListResponse,
    SearchResponse,
    SearchResult,
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


def _safe_attachment_segment(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]", "_", str(value or "").strip())
    return clean.strip(".") or "default"


def _attachment_from_path(workspace: Path | None, raw_path: str) -> ChatAttachment:
    text = str(raw_path or "").strip()
    path_obj = Path(text)
    name = path_obj.name or text
    mime, _ = mimetypes.guess_type(name)
    media_type = mime or "application/octet-stream"
    is_image = media_type.startswith("image/")
    rel_path = text
    url = ""
    if workspace is not None and text:
        root = workspace.expanduser().resolve()
        candidate = (root / path_obj).resolve() if not path_obj.is_absolute() else path_obj.resolve()
        try:
            rel = candidate.relative_to(root)
            rel_path = rel.as_posix()
            parts = rel.parts
            if len(parts) == 3 and parts[0] == "chat_attachments":
                url = f"/v1/chat/attachments/{parts[1]}/{parts[2]}"
        except ValueError:
            pass
    return ChatAttachment(
        path=rel_path,
        name=name,
        media_type=media_type,
        is_image=is_image,
        url=url,
    )


def _chat_response(
    *, runtime: Any, request_id: str, session_id: str, outbound: Any, started: float
) -> ChatResponse:
    metadata = getattr(outbound, "metadata", None) or {}
    return ChatResponse(
        request_id=request_id,
        trace_id=str(metadata.get("trace_id", "")),
        session_id=session_id,
        answer=outbound.content,
        reasoning_content=str(metadata.get("reasoning_content", "") or ""),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        trace=_flatten_trace(metadata),
        pending_confirmation=_pending_for_session(runtime, session_id),
    )


def _task_summary(record: Any) -> TaskRunSummary:
    return TaskRunSummary(
        trace_id=record.trace_id,
        session_id=record.session_key.split(":", 1)[-1],
        status=record.status,
        phase=record.phase,
        iteration=record.iteration,
        tool_chain=list(record.tool_chain),
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
        completed_at=record.completed_at,
    )


def _knowledge_document(document: Any) -> KnowledgeDocument:
    return KnowledgeDocument(
        id=document.id,
        source_path=document.source_path,
        title=document.title,
        media_type=document.media_type,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        updated_at=document.updated_at,
        vector_status=document.vector_status,
        vector_error=document.vector_error,
    )


def _dashboard_sessions(manager: Any) -> list[dict[str, Any]]:
    """Return dashboard sessions while exposing at most one blank session."""
    sessions = manager.list_dashboard_sessions(channel="api", limit=200)
    result: list[dict[str, Any]] = []
    blank_seen = False
    for item in sessions:
        if int(item.get("message_count", 0) or 0) == 0:
            if blank_seen:
                continue
            blank_seen = True
        result.append(item)
    return result


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
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
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

    @application.get(
        "/v1/usage",
        tags=["system"],
        dependencies=[Depends(verify_api_key)],
    )
    async def usage(request: Request) -> dict[str, Any]:
        """Return persisted model call, token and estimated cost totals."""
        store = _reliability_store(request.app.state.runtime)
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Usage store is unavailable",
            )
        return store.model_usage_summary()

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

    @application.get(
        "/v1/knowledge/documents",
        response_model=KnowledgeDocumentListResponse,
        tags=["rag"],
        dependencies=[Depends(verify_api_key)],
    )
    async def knowledge_documents(request: Request) -> KnowledgeDocumentListResponse:
        service = getattr(request.app.state.runtime, "rag_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="RAG service is unavailable")
        return KnowledgeDocumentListResponse(
            documents=[_knowledge_document(item) for item in service.list_documents()]
        )

    @application.post(
        "/v1/knowledge/upload",
        response_model=KnowledgeDocument,
        tags=["rag"],
        dependencies=[Depends(verify_api_key)],
    )
    async def upload_knowledge(
        request: Request,
        file: UploadFile = File(...),
        index: bool = Form(True),
    ) -> KnowledgeDocument:
        service = getattr(request.app.state.runtime, "rag_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="RAG service is unavailable")
        raw_name = Path(file.filename or "upload").name
        if not raw_name or raw_name in {".", ".."}:
            raise HTTPException(status_code=400, detail="文件名无效")
        if len(raw_name) > 160:
            raise HTTPException(status_code=400, detail="文件名不能超过 160 个字符")
        suffix = Path(raw_name).suffix.lower()
        allowed = {".txt", ".md", ".markdown", ".json", ".yaml", ".yml", ".toml", ".py", ".js", ".ts", ".tsx", ".jsx", ".csv", ".pdf"}
        if suffix not in allowed:
            raise HTTPException(status_code=400, detail="暂不支持该文件类型，请上传文本、代码或 PDF")
        upload_dir = Path(request.app.state.runtime.workspace) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / raw_name
        if target.exists():
            target = upload_dir / f"{target.stem}-{uuid.uuid4().hex[:8]}{target.suffix}"
        size = 0
        try:
            with target.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 20 * 1024 * 1024:
                        raise HTTPException(status_code=413, detail="单个文件不能超过 20 MB")
                    handle.write(chunk)
            if index:
                document = await service.ingest_file(target.relative_to(service.workspace).as_posix())
            else:
                raise HTTPException(status_code=400, detail="当前上传接口必须建立索引")
            stored = service.store.get_document(document.document_id)
            if stored is None:
                raise HTTPException(status_code=500, detail="知识库索引写入后无法读取文档")
            return _knowledge_document(stored)
        except HTTPException:
            target.unlink(missing_ok=True)
            raise
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"文件处理失败：{exc}") from exc

    @application.delete(
        "/v1/knowledge/documents/{document_id}",
        tags=["rag"],
        dependencies=[Depends(verify_api_key)],
    )
    async def delete_knowledge_document(document_id: str, request: Request) -> dict[str, Any]:
        service = getattr(request.app.state.runtime, "rag_service", None)
        if service is None or not hasattr(service, "delete_document"):
            raise HTTPException(status_code=503, detail="RAG service is unavailable")
        if not service.delete_document(document_id):
            raise HTTPException(status_code=404, detail="知识库文档不存在")
        return {"deleted_document_id": document_id}

    @application.get(
        "/v1/search",
        response_model=SearchResponse,
        tags=["chat"],
        dependencies=[Depends(verify_api_key)],
    )
    async def search_sessions(request: Request, q: str = Query(min_length=1, max_length=200), limit: int = 20) -> SearchResponse:
        manager = getattr(request.app.state.runtime, "session_manager", None)
        if manager is None or not hasattr(manager, "search_messages"):
            raise HTTPException(status_code=503, detail="Session store is unavailable")
        rows, total = manager.search_messages(q, channel="api", limit=max(1, min(limit, 50)))
        return SearchResponse(
            query=q,
            total=total,
            results=[SearchResult(**row) for row in rows],
        )

    @application.post(
        "/v1/chat/attachments",
        response_model=ChatAttachment,
        tags=["chat"],
        dependencies=[Depends(verify_api_key)],
    )
    async def upload_chat_attachment(
        request: Request,
        file: UploadFile = File(...),
        session_id: str = Form("default"),
    ) -> ChatAttachment:
        workspace = Path(request.app.state.runtime.workspace).expanduser().resolve()
        raw_name = Path(file.filename or "attachment").name
        if not raw_name or raw_name in {".", ".."}:
            raise HTTPException(status_code=400, detail="文件名无效")
        if len(raw_name) > 160:
            raise HTTPException(status_code=400, detail="文件名不能超过 160 个字符")
        suffix = Path(raw_name).suffix.lower()
        allowed = {
            ".png", ".jpg", ".jpeg", ".webp", ".gif",
            ".txt", ".md", ".markdown", ".json", ".yaml", ".yml", ".toml",
            ".py", ".js", ".ts", ".tsx", ".jsx", ".csv", ".pdf",
        }
        if suffix not in allowed:
            raise HTTPException(status_code=400, detail="暂不支持该附件类型，请上传图片、文本、代码或 PDF")
        safe_session = _safe_attachment_segment(session_id)
        attach_dir = workspace / "chat_attachments" / safe_session
        attach_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = _safe_attachment_segment(Path(raw_name).stem)[:80]
        stored_name = f"{safe_stem}-{uuid.uuid4().hex[:8]}{suffix}"
        target = attach_dir / stored_name
        size = 0
        try:
            with target.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 15 * 1024 * 1024:
                        raise HTTPException(status_code=413, detail="单个附件不能超过 15 MB")
                    handle.write(chunk)
        except HTTPException:
            target.unlink(missing_ok=True)
            raise
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"附件保存失败：{exc}") from exc

        rel = target.relative_to(workspace).as_posix()
        mime, _ = mimetypes.guess_type(raw_name)
        media_type = mime or file.content_type or "application/octet-stream"
        return ChatAttachment(
            path=rel,
            name=raw_name,
            media_type=media_type,
            is_image=media_type.startswith("image/"),
            url=f"/v1/chat/attachments/{safe_session}/{stored_name}",
        )

    @application.get(
        "/v1/chat/attachments/{session_dir}/{filename}",
        tags=["chat"],
    )
    async def get_chat_attachment(
        session_dir: str,
        filename: str,
        request: Request,
    ) -> FileResponse:
        workspace = Path(request.app.state.runtime.workspace).expanduser().resolve()
        base = (workspace / "chat_attachments").resolve()
        candidate = (base / session_dir / filename).resolve()
        if not candidate.is_file() or not candidate.is_relative_to(base):
            raise HTTPException(status_code=404, detail="Attachment not found")
        return FileResponse(candidate)

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
            request_kwargs: dict[str, Any] = {
                "session_id": payload.session_id,
                "sender_id": payload.sender_id,
                "content": payload.message,
                "timeout": payload.timeout_seconds,
            }
            if payload.mode != "balanced":
                request_kwargs["mode"] = payload.mode
            if payload.media:
                request_kwargs["media"] = list(payload.media)
            request_id, outbound = await runtime.api_channel.request(**request_kwargs)
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
                for item in _dashboard_sessions(manager)
            ]
        )

    @application.get(
        "/v1/sessions/{session_id}/tasks",
        response_model=TaskRunListResponse,
        tags=["tasks"],
        dependencies=[Depends(verify_api_key)],
    )
    async def list_tasks(
        session_id: str,
        request: Request,
        limit: int = 50,
    ) -> TaskRunListResponse:
        store = _reliability_store(request.app.state.runtime)
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Task journal is unavailable",
            )
        records = store.list_task_runs(
            session_key=f"api:{session_id}",
            limit=max(1, min(limit, 500)),
        )
        return TaskRunListResponse(tasks=[_task_summary(record) for record in records])

    @application.get(
        "/v1/tasks/{trace_id}",
        response_model=TaskRunDetailResponse,
        tags=["tasks"],
        dependencies=[Depends(verify_api_key)],
    )
    async def task_detail(
        trace_id: str,
        request: Request,
        session_id: str = Query(min_length=1, max_length=128),
    ) -> TaskRunDetailResponse:
        store = _reliability_store(request.app.state.runtime)
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Task journal is unavailable",
            )
        record = store.get_task_run(trace_id)
        if record is None or record.session_key != f"api:{session_id}":
            raise HTTPException(status_code=404, detail="Task trace not found")
        summary = _task_summary(record)
        return TaskRunDetailResponse(
            **summary.model_dump(),
            user_content=record.user_content,
            checkpoint=dict(record.checkpoint),
        )

    @application.post(
        "/v1/sessions/{session_id}/interrupt",
        tags=["tasks"],
        dependencies=[Depends(verify_api_key)],
    )
    async def interrupt_session(session_id: str, request: Request) -> dict[str, Any]:
        loop = getattr(request.app.state.runtime, "agent_loop", None)
        if loop is None:
            raise HTTPException(status_code=503, detail="Agent loop is unavailable")
        result = loop.request_interrupt(f"api:{session_id}", sender="dashboard", command="/stop")
        return {"status": result.status, "message": result.message, "session_id": session_id}

    @application.post(
        "/v1/tasks/{trace_id}/resume",
        response_model=ChatResponse,
        tags=["tasks"],
        dependencies=[Depends(verify_api_key)],
    )
    async def resume_task(
        trace_id: str,
        request: Request,
        session_id: str = Query(min_length=1, max_length=128),
        timeout_seconds: float = Query(default=120.0, ge=1.0, le=300.0),
        mode: str = Query(default="balanced", pattern="^(fast|balanced|deep)$"),
    ) -> ChatResponse:
        runtime = request.app.state.runtime
        store = _reliability_store(runtime)
        record = store.get_task_run(trace_id) if store is not None else None
        if record is None or record.session_key != f"api:{session_id}":
            raise HTTPException(status_code=404, detail="Task trace not found")
        if record.status not in {TASK_CANCELLED, TASK_FAILED, TASK_INTERRUPTED, TASK_NEEDS_REVIEW}:
            raise HTTPException(status_code=409, detail="当前任务状态不支持重试")
        started = time.perf_counter()
        try:
            request_id, outbound = await runtime.api_channel.request(
                session_id=session_id,
                sender_id="dashboard",
                content=record.user_content,
                timeout=timeout_seconds,
                mode=mode,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise HTTPException(status_code=504, detail="Agent response timeout") from exc
        return _chat_response(runtime=runtime, request_id=request_id, session_id=session_id, outbound=outbound, started=started)

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

        # An empty session is already a usable new conversation. Reuse it
        # instead of creating another blank row on every click or from
        # multiple browser tabs.
        existing_empty = next(
            (
                item
                for item in _dashboard_sessions(manager)
                if int(item.get("message_count", 0) or 0) == 0
            ),
            None,
        )
        if existing_empty is not None:
            return SessionCreateResponse(session=SessionSummary(**existing_empty))

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

    @application.delete(
        "/v1/sessions/{session_id}",
        response_model=SessionDeleteResponse,
        tags=["chat"],
        dependencies=[Depends(verify_api_key)],
    )
    async def delete_session(session_id: str, request: Request) -> SessionDeleteResponse:
        manager = getattr(request.app.state.runtime, "session_manager", None)
        if manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session store is unavailable",
            )

        session_key = f"api:{session_id}"
        if not manager.session_exists(session_key):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        manager.delete_session(session_key)

        # Keep exactly one usable blank session available after deletion.
        sessions = _dashboard_sessions(manager)
        replacement = next(
            (item for item in sessions if int(item.get("message_count", 0) or 0) == 0),
            None,
        )
        if replacement is None:
            replacement_id = f"web-{uuid.uuid4()}"
            created = manager.create_session(
                f"api:{replacement_id}",
                metadata={"source": "dashboard"},
            )
            replacement = {
                "session_id": replacement_id,
                "title": "新会话",
                "preview": "",
                "created_at": created.created_at.isoformat(),
                "updated_at": created.updated_at.isoformat(),
                "message_count": 0,
            }

        return SessionDeleteResponse(
            deleted_session_id=session_id,
            replacement_session=SessionSummary(**replacement),
        )

    @application.patch(
        "/v1/sessions/{session_id}",
        response_model=SessionSummary,
        tags=["chat"],
        dependencies=[Depends(verify_api_key)],
    )
    async def update_session(
        session_id: str,
        payload: SessionUpdateRequest,
        request: Request,
    ) -> SessionSummary:
        manager = getattr(request.app.state.runtime, "session_manager", None)
        if manager is None or not hasattr(manager, "update_session_metadata"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session store is unavailable",
            )
        session_key = f"api:{session_id}"
        if not manager.session_exists(session_key):
            raise HTTPException(status_code=404, detail="Session not found")
        manager.update_session_metadata(
            session_key,
            title=payload.title,
            pinned=payload.pinned,
        )
        matched = next(
            (
                item
                for item in _dashboard_sessions(manager)
                if item.get("session_id") == session_id
            ),
            None,
        )
        if matched is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionSummary(**matched)

    @application.post(
        "/v1/sessions/{session_id}/truncate",
        response_model=SessionTruncateResponse,
        tags=["chat"],
        dependencies=[Depends(verify_api_key)],
    )
    async def truncate_session(
        session_id: str,
        payload: SessionTruncateRequest,
        request: Request,
    ) -> SessionTruncateResponse:
        manager = getattr(request.app.state.runtime, "session_manager", None)
        if manager is None or not hasattr(manager, "truncate_from_seq"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session store is unavailable",
            )
        session_key = f"api:{session_id}"
        if not manager.session_exists(session_key):
            raise HTTPException(status_code=404, detail="Session not found")
        deleted = manager.truncate_from_seq(session_key, payload.from_seq)
        return SessionTruncateResponse(
            session_id=session_id,
            deleted_count=deleted,
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
                stream_kwargs: dict[str, Any] = {
                    "session_id": payload.session_id,
                    "sender_id": payload.sender_id,
                    "content": payload.message,
                    "timeout": payload.timeout_seconds,
                }
                if payload.mode != "balanced":
                    stream_kwargs["mode"] = payload.mode
                if payload.media:
                    stream_kwargs["media"] = list(payload.media)
                async for request_id, event in runtime.api_channel.request_stream(**stream_kwargs):
                    if await request.is_disconnected():
                        loop = getattr(runtime, "agent_loop", None)
                        if loop is not None:
                            loop.request_interrupt(
                                f"api:{payload.session_id}",
                                sender="client_disconnect",
                                command="/disconnect",
                            )
                        return
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
            except asyncio.CancelledError:
                loop = getattr(runtime, "agent_loop", None)
                if loop is not None:
                    loop.request_interrupt(
                        f"api:{payload.session_id}",
                        sender="client_disconnect",
                        command="/disconnect",
                    )
                raise
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
        workspace = getattr(request.app.state.runtime, "workspace", None)
        workspace_path = Path(workspace) if workspace is not None else None
        messages = []
        for idx, item in enumerate(session.messages):
            role = str(item.get("role", ""))
            if role not in {"user", "assistant"}:
                continue
            raw_media = item.get("media") or []
            media_items = [
                _attachment_from_path(workspace_path, str(path))
                for path in raw_media
                if path
            ]
            messages.append(
                SessionMessage(
                    seq=int(item.get("seq", idx) or 0),
                    role=role,
                    content=str(item.get("content", "")),
                    reasoning_content=str(item.get("reasoning_content", "") or ""),
                    timestamp=str(item.get("timestamp", "")),
                    media=media_items,
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

    @application.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> Any:
        if full_path.startswith(("v1/", "health", "ready", "docs", "openapi.json", "redoc")):
            raise HTTPException(status_code=404, detail="Not Found")
        if full_path:
            candidate = (frontend_dist / full_path).resolve()
            if (
                candidate.is_file()
                and candidate.is_relative_to(frontend_dist.resolve())
            ):
                return FileResponse(candidate)
        index_file = frontend_dist / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Not Found")

    return application


app = create_app()
