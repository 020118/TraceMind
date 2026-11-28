export interface ToolTraceItem {
  call_id: string;
  name: string;
  arguments: Record<string, unknown>;
  result: string;
  status: string;
  duration_ms: number;
  operation_id: string;
}

export interface ChatAttachment {
  path: string;
  name: string;
  media_type: string;
  is_image: boolean;
  url: string;
}

export interface ConfirmationDetail {
  label: string;
  value: string;
}

export interface ConfirmationSpec {
  title?: string;
  summary?: string;
  severity?: "info" | "warning" | "danger";
  confirm_label?: string;
  cancel_label?: string;
  details?: ConfirmationDetail[];
}

export interface PendingConfirmation {
  operation_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  status: string;
  created_at: string;
  expires_at: string | null;
  spec?: ConfirmationSpec;
}

export interface ChatResponse {
  request_id: string;
  trace_id: string;
  session_id: string;
  answer: string;
  reasoning_content?: string;
  elapsed_ms: number;
  trace: ToolTraceItem[];
  pending_confirmation: PendingConfirmation | null;
}

export interface TaskRunSummary {
  trace_id: string;
  session_id: string;
  status: string;
  phase: string;
  iteration: number;
  tool_chain: Array<Record<string, unknown>>;
  error: string;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface TaskRunDetail extends TaskRunSummary {
  user_content: string;
  checkpoint: Record<string, unknown>;
}

export interface SessionMessage {
  seq?: number;
  role: "user" | "assistant";
  content: string;
  reasoning_content?: string;
  timestamp: string;
  media?: ChatAttachment[];
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
  pinned?: boolean;
}

export interface SessionListResponse {
  sessions: SessionSummary[];
}

export interface SessionCreateResponse {
  session: SessionSummary;
}

export interface SessionDeleteResponse {
  deleted_session_id: string;
  replacement_session: SessionSummary;
}

export interface SessionTruncateResponse {
  session_id: string;
  deleted_count: number;
}

export interface ModelUsageItem {
  model: string;
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
}

export interface UsageSummary extends Omit<ModelUsageItem, "model"> {
  by_model: ModelUsageItem[];
}

export interface KnowledgeDocument {
  id: string;
  source_path: string;
  title: string;
  media_type: string;
  chunk_count: number;
  created_at: string;
  updated_at: string;
  vector_status: string;
  vector_error: string;
}

export interface SearchResult {
  session_id: string;
  role: string;
  content: string;
  timestamp: string;
  message_id: string;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total: number;
}

export interface ChatMessage extends SessionMessage {
  id: string;
  elapsedMs?: number;
  traceId?: string;
  streamStatus?: string;
  isThinking?: boolean;
}

export interface ChatStreamEvent {
  type:
    | "turn_started"
    | "model_started"
    | "thinking_started"
    | "thinking_delta"
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
  trace_id?: string;
  message?: string;
  response?: ChatResponse;
}
