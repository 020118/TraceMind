export interface ToolTraceItem {
  call_id: string;
  name: string;
  arguments: Record<string, unknown>;
  result: string;
  status: string;
  duration_ms: number;
  operation_id: string;
}

export interface PendingConfirmation {
  operation_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  status: string;
  created_at: string;
  expires_at: string | null;
}

export interface ChatResponse {
  request_id: string;
  session_id: string;
  answer: string;
  elapsed_ms: number;
  trace: ToolTraceItem[];
  pending_confirmation: PendingConfirmation | null;
}

export interface SessionMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  trace: ToolTraceItem[];
}

export interface SessionHistoryResponse {
  session_id: string;
  messages: SessionMessage[];
  pending_confirmation: PendingConfirmation | null;
}

export interface SessionSummary {
  session_id: string;
  title: string;
  preview: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface SessionListResponse {
  sessions: SessionSummary[];
}

export interface SessionCreateResponse {
  session: SessionSummary;
}

export interface ChatMessage extends SessionMessage {
  id: string;
  elapsedMs?: number;
  streamStatus?: string;
}

export interface ChatStreamEvent {
  type:
    | "turn_started"
    | "model_started"
    | "answer_started"
    | "answer_delta"
    | "answer_reset"
    | "tool_started"
    | "tool_finished"
    | "turn_finished"
    | "error";
  iteration?: number;
  delta?: string;
  call_id?: string;
  name?: string;
  arguments?: Record<string, unknown>;
  result?: string;
  status?: string;
  duration_ms?: number;
  operation_id?: string;
  message?: string;
  response?: ChatResponse;
}
