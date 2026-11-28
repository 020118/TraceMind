import type {
  ChatAttachment,
  ChatResponse,
  ChatStreamEvent,
  KnowledgeDocument,
  SearchResponse,
  SessionCreateResponse,
  SessionDeleteResponse,
  SessionHistoryResponse,
  SessionListResponse,
  SessionSummary,
  SessionTruncateResponse,
  TaskRunDetail,
  TaskRunSummary,
  UsageSummary
} from "./types";

const ENV_API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
export const API_BASE_STORAGE_KEY = "tracemind.api-base-url";

export function getApiBaseUrl(): string {
  try {
    const saved = localStorage.getItem(API_BASE_STORAGE_KEY);
    if (saved !== null) {
      return saved.trim().replace(/\/$/, "");
    }
  } catch {
    // ignore storage errors
  }
  return ENV_API_BASE;
}

export function setApiBaseUrl(value: string): string {
  const clean = value.trim().replace(/\/$/, "");
  try {
    if (clean) {
      localStorage.setItem(API_BASE_STORAGE_KEY, clean);
    } else {
      localStorage.removeItem(API_BASE_STORAGE_KEY);
    }
  } catch {
    // ignore storage errors
  }
  return clean;
}

export function resolveAssetUrl(url: string): string {
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://") || url.startsWith("data:")) {
    return url;
  }
  return `${getApiBaseUrl()}${url}`;
}

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
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
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
  apiKey: string,
  mode = "balanced",
  media: string[] = []
): Promise<ChatResponse> {
  return request<ChatResponse>("/v1/chat", apiKey, {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      sender_id: "dashboard",
      message,
      mode,
      media,
      timeout_seconds: 120
    })
  });
}

export async function* streamChat(
  sessionId: string,
  message: string,
  apiKey: string,
  mode = "balanced",
  signal?: AbortSignal,
  media: string[] = []
): AsyncGenerator<ChatStreamEvent> {
  const response = await fetch(`${getApiBaseUrl()}/v1/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(apiKey ? { "X-API-Key": apiKey } : {})
    },
    signal,
    body: JSON.stringify({
      session_id: sessionId,
      sender_id: "dashboard",
      message,
      mode,
      media,
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

export async function uploadChatAttachment(
  sessionId: string,
  file: File,
  apiKey: string
): Promise<ChatAttachment> {
  const form = new FormData();
  form.append("file", file);
  form.append("session_id", sessionId || "default");
  const response = await fetch(`${getApiBaseUrl()}/v1/chat/attachments`, {
    method: "POST",
    headers: apiKey ? { "X-API-Key": apiKey } : {},
    body: form
  });
  if (!response.ok) {
    let message = `附件上传失败（HTTP ${response.status}）`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (payload.detail) {
        message = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
      }
    } catch {
      // ignore
    }
    throw new ApiError(message, response.status);
  }
  return response.json() as Promise<ChatAttachment>;
}

export function patchSession(
  sessionId: string,
  payload: { title?: string; pinned?: boolean },
  apiKey: string
): Promise<SessionSummary> {
  return request<SessionSummary>(
    `/v1/sessions/${encodeURIComponent(sessionId)}`,
    apiKey,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    }
  );
}

export function truncateSession(
  sessionId: string,
  fromSeq: number,
  apiKey: string
): Promise<SessionTruncateResponse> {
  return request<SessionTruncateResponse>(
    `/v1/sessions/${encodeURIComponent(sessionId)}/truncate`,
    apiKey,
    {
      method: "POST",
      body: JSON.stringify({ from_seq: fromSeq })
    }
  );
}

export function interruptSession(sessionId: string, apiKey: string): Promise<{ status: string; message: string }> {
  return request<{ status: string; message: string }>(
    `/v1/sessions/${encodeURIComponent(sessionId)}/interrupt`, apiKey, { method: "POST" }
  );
}

export function listKnowledge(apiKey: string): Promise<{ documents: KnowledgeDocument[] }> {
  return request<{ documents: KnowledgeDocument[] }>("/v1/knowledge/documents", apiKey);
}

export async function uploadKnowledge(file: File, apiKey: string): Promise<KnowledgeDocument> {
  const form = new FormData();
  form.append("file", file);
  form.append("index", "true");
  const response = await fetch(`${getApiBaseUrl()}/v1/knowledge/upload`, {
    method: "POST",
    headers: apiKey ? { "X-API-Key": apiKey } : {},
    body: form
  });
  if (!response.ok) throw new ApiError(`上传失败（HTTP ${response.status}）`, response.status);
  return response.json() as Promise<KnowledgeDocument>;
}

export function deleteKnowledge(documentId: string, apiKey: string): Promise<{ deleted_document_id: string }> {
  return request<{ deleted_document_id: string }>(`/v1/knowledge/documents/${encodeURIComponent(documentId)}`, apiKey, { method: "DELETE" });
}

export function resumeTask(traceId: string, sessionId: string, apiKey: string, mode = "balanced"): Promise<ChatResponse> {
  return request<ChatResponse>(
    `/v1/tasks/${encodeURIComponent(traceId)}/resume?session_id=${encodeURIComponent(sessionId)}&mode=${encodeURIComponent(mode)}`,
    apiKey,
    { method: "POST" }
  );
}

export function searchSessions(query: string, apiKey: string): Promise<SearchResponse> {
  return request<SearchResponse>(`/v1/search?q=${encodeURIComponent(query)}&limit=30`, apiKey);
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

export function getUsage(apiKey: string): Promise<UsageSummary> {
  return request<UsageSummary>("/v1/usage", apiKey);
}

export function createSession(apiKey: string): Promise<SessionCreateResponse> {
  return request<SessionCreateResponse>("/v1/sessions", apiKey, {
    method: "POST",
    body: "{}"
  });
}

export function deleteSession(
  sessionId: string,
  apiKey: string
): Promise<SessionDeleteResponse> {
  return request<SessionDeleteResponse>(
    `/v1/sessions/${encodeURIComponent(sessionId)}`,
    apiKey,
    { method: "DELETE" }
  );
}

export function listTasks(
  sessionId: string,
  apiKey: string,
  limit = 50
): Promise<{ tasks: TaskRunSummary[] }> {
  return request<{ tasks: TaskRunSummary[] }>(
    `/v1/sessions/${encodeURIComponent(sessionId)}/tasks?limit=${limit}`,
    apiKey
  );
}

export function getTask(
  traceId: string,
  sessionId: string,
  apiKey: string
): Promise<TaskRunDetail> {
  return request<TaskRunDetail>(
    `/v1/tasks/${encodeURIComponent(traceId)}?session_id=${encodeURIComponent(sessionId)}`,
    apiKey
  );
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
