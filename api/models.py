from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=20_000)
    sender_id: str | None = Field(default=None, max_length=128)
    timeout_seconds: float = Field(default=120.0, ge=1.0, le=300.0)


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
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


class OperationDecisionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    timeout_seconds: float = Field(default=120.0, ge=1.0, le=300.0)


class SessionMessage(BaseModel):
    role: str
    content: str
    timestamp: str = ""
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


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary] = Field(default_factory=list)


class SessionCreateResponse(BaseModel):
    session: SessionSummary


class HealthResponse(BaseModel):
    status: str

