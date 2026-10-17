import type {
  ChatResponse,
  ChatStreamEvent,
  SessionCreateResponse,
  SessionHistoryResponse,
  SessionListResponse
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message);
  }
}

async function request<T>(
  path: string,
  apiKey: string,
  init?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(apiKey ? { "X-API-Key": apiKey } : {}),
      ...init?.headers
    }
  });
  if (!response.ok) {
    let message = `请求失败（HTTP ${response.status}）`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (payload.detail) {
        message = typeof payload.detail === "string"
          ? payload.detail
          : JSON.stringify(payload.detail);
      }
    } catch {
      // Keep the HTTP fallback message when the response is not JSON.
    }
    throw new ApiError(message, response.status);
  }
  return response.json() as Promise<T>;
}

export function sendChat(
  sessionId: string,
  message: string,
  apiKey: string
): Promise<ChatResponse> {
  return request<ChatResponse>("/v1/chat", apiKey, {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      sender_id: "dashboard",
      message,
      timeout_seconds: 120
    })
  });
}

export async function* streamChat(
  sessionId: string,
  message: string,
  apiKey: string
): AsyncGenerator<ChatStreamEvent> {
  const response = await fetch(`${API_BASE}/v1/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(apiKey ? { "X-API-Key": apiKey } : {})
    },
    body: JSON.stringify({
      session_id: sessionId,
      sender_id: "dashboard",
      message,
      timeout_seconds: 120
    })
  });

  if (!response.ok) {
    let errorMessage = `请求失败（HTTP ${response.status}）`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (payload.detail) {
        errorMessage = typeof payload.detail === "string"
          ? payload.detail
          : JSON.stringify(payload.detail);
      }
    } catch {
      // Keep the HTTP fallback message when the response is not JSON.
    }
    throw new ApiError(errorMessage, response.status);
  }
  if (!response.body) {
    throw new ApiError("浏览器没有收到流式响应体", response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        const clean = line.trim();
        if (clean) yield JSON.parse(clean) as ChatStreamEvent;
      }
      if (done) break;
    }
    if (buffer.trim()) {
      yield JSON.parse(buffer) as ChatStreamEvent;
    }
  } finally {
    reader.releaseLock();
  }
}

export function getHistory(
  sessionId: string,
  apiKey: string
): Promise<SessionHistoryResponse> {
  return request<SessionHistoryResponse>(
    `/v1/sessions/${encodeURIComponent(sessionId)}/messages`,
    apiKey
  );
}

export function listSessions(apiKey: string): Promise<SessionListResponse> {
  return request<SessionListResponse>("/v1/sessions", apiKey);
}

export function createSession(apiKey: string): Promise<SessionCreateResponse> {
  return request<SessionCreateResponse>("/v1/sessions", apiKey, {
    method: "POST",
    body: "{}"
  });
}

export function decideOperation(
  operationId: string,
  decision: "confirm" | "cancel",
  sessionId: string,
  apiKey: string
): Promise<ChatResponse> {
  return request<ChatResponse>(
    `/v1/operations/${encodeURIComponent(operationId)}/${decision}`,
    apiKey,
    {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, timeout_seconds: 120 })
    }
  );
}
