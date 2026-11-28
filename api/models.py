from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=20_000)
    sender_id: str | None = Field(default=None, max_length=128)
    timeout_seconds: float = Field(default=120.0, ge=1.0, le=300.0)
    mode: str = Field(default="balanced", pattern="^(fast|balanced|deep)$")
    media: list[str] = Field(default_factory=list, max_length=8)


class ChatResponse(BaseModel):
    request_id: str
    trace_id: str = ""
    session_id: str
    answer: str
    reasoning_content: str = ""
    elapsed_ms: int
    trace: list["ToolTraceItem"] = Field(default_factory=list)
    pending_confirmation: "PendingConfirmation | None" = None


class ToolTraceItem(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    status: str = "succeeded"
    duration_ms: int = 0
    operation_id: str = ""


class PendingConfirmation(BaseModel):
    operation_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: str
    created_at: str
    expires_at: str | None = None
    spec: dict[str, Any] = Field(default_factory=dict)


class OperationDecisionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    timeout_seconds: float = Field(default=120.0, ge=1.0, le=300.0)


class OperationDecideRequest(OperationDecisionRequest):
    decision: str = Field(pattern="^(confirm|cancel)$")


class ChatAttachment(BaseModel):
    path: str
    name: str
    media_type: str = ""
    is_image: bool = False
    url: str = ""


class SessionMessage(BaseModel):
    seq: int = 0
    role: str
    content: str
    reasoning_content: str = ""
    timestamp: str = ""
    media: list[ChatAttachment] = Field(default_factory=list)
    trace: list[ToolTraceItem] = Field(default_factory=list)


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[SessionMessage] = Field(default_factory=list)
    pending_confirmation: PendingConfirmation | None = None


class SessionSummary(BaseModel):
    session_id: str
    title: str
    preview: str = ""
    created_at: str
    updated_at: str
    message_count: int = 0
    pinned: bool = False


class SessionUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    pinned: bool | None = None


class SessionTruncateRequest(BaseModel):
    from_seq: int = Field(ge=0)


class SessionTruncateResponse(BaseModel):
    session_id: str
    deleted_count: int


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary] = Field(default_factory=list)


class TaskRunSummary(BaseModel):
    trace_id: str
    session_id: str
    status: str
    phase: str
    iteration: int = 0
    tool_chain: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""
    created_at: str
    updated_at: str
    completed_at: str | None = None


class TaskRunListResponse(BaseModel):
    tasks: list[TaskRunSummary] = Field(default_factory=list)


class TaskRunDetailResponse(TaskRunSummary):
    user_content: str
    checkpoint: dict[str, Any] = Field(default_factory=dict)


class TaskResumeResponse(BaseModel):
    task: TaskRunSummary
    response: ChatResponse


class KnowledgeDocument(BaseModel):
    id: str
    source_path: str
    title: str
    media_type: str
    chunk_count: int
    created_at: str = ""
    updated_at: str = ""
    vector_status: str = ""
    vector_error: str = ""


class KnowledgeDocumentListResponse(BaseModel):
    documents: list[KnowledgeDocument] = Field(default_factory=list)


class SearchResult(BaseModel):
    session_id: str
    role: str
    content: str
    timestamp: str = ""
    message_id: str = ""


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult] = Field(default_factory=list)
    total: int = 0


class SessionCreateResponse(BaseModel):
    session: SessionSummary


class SessionDeleteResponse(BaseModel):
    deleted_session_id: str
    replacement_session: SessionSummary


class HealthResponse(BaseModel):
    status: str

