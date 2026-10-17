import {
  FormEvent,
  KeyboardEvent,
  useEffect,
  useRef,
  useState
} from "react";
import {
  ApiError,
  createSession,
  decideOperation,
  getHistory,
  listSessions,
  streamChat
} from "./api";
import type {
  ChatMessage,
  ChatResponse,
  PendingConfirmation,
  SessionSummary,
  ToolTraceItem
} from "./types";

const WELCOME: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "你好，我是你的个人研究与代码助理。你可以让我阅读项目、修改代码、运行检查，或整理研究资料。",
  timestamp: "",
  trace: []
};

const AGENT_AVATAR_SRC = "/tracemind-avatar.jpg";
const SESSION_ID_STORAGE_KEY = "tracemind.session-id";
const API_KEY_STORAGE_KEY = "tracemind.api-key";

function AgentAvatar() {
  return <img src={AGENT_AVATAR_SRC} alt="" />;
}

function newSessionId(): string {
  return `web-${crypto.randomUUID()}`;
}

function storedSessionId(): string {
  return localStorage.getItem(SESSION_ID_STORAGE_KEY)
    || localStorage.getItem("myagent.session-id")
    || "";
}

function toMessage(
  role: "user" | "assistant",
  content: string,
  trace: ToolTraceItem[] = [],
  elapsedMs?: number
): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    timestamp: new Date().toISOString(),
    trace,
    elapsedMs
  };
}

function timeLabel(value: string): string {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "";
  return parsed.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit"
  });
}

function sessionTimeLabel(value: string): string {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "";
  const now = new Date();
  if (parsed.toDateString() === now.toDateString()) {
    return parsed.toLocaleTimeString("zh-CN", {
      hour: "2-digit",
      minute: "2-digit"
    });
  }
  return parsed.toLocaleDateString("zh-CN", {
    month: "2-digit",
    day: "2-digit"
  });
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    succeeded: "完成",
    failed: "失败",
    timeout: "超时",
    awaiting_confirmation: "等待确认",
    cancelled: "已取消",
    running: "执行中"
  };
  return labels[status.toLowerCase()] || status;
}

function toolLabel(name: string): string {
  const labels: Record<string, string> = {
    tool_search: "发现可用工具",
    read_file: "读取文件",
    list_dir: "浏览目录",
    write_file: "写入文件",
    edit_file: "修改文件",
    delete_file: "删除文件",
    python_syntax_check: "Python 语法检查",
    run_pytest: "运行测试",
    web_search: "搜索网络",
    web_fetch: "读取网页"
  };
  return labels[name] || name;
}

function shortResult(result: string): string {
  const clean = result.trim();
  if (clean.length <= 180) return clean;
  return `${clean.slice(0, 180)}…`;
}

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

export default function App() {
  const [sessionId, setSessionId] = useState(storedSessionId);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [apiKey, setApiKey] = useState(
    () => sessionStorage.getItem(API_KEY_STORAGE_KEY)
      || sessionStorage.getItem("myagent.api-key")
      || ""
  );
  const [historyVersion, setHistoryVersion] = useState(0);
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [pending, setPending] = useState<PendingConfirmation | null>(null);
  const [error, setError] = useState("");
  const [serviceStatus, setServiceStatus] = useState<"checking" | "ready" | "offline">("checking");
  const messageListRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    sessionStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
  }, [apiKey]);

  useEffect(() => {
    const list = messageListRef.current;
    if (list) list.scrollTo({ top: list.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  useEffect(() => {
    let cancelled = false;
    listSessions(apiKey)
      .then((response) => {
        if (cancelled) return;
        setSessions(response.sessions);
        setSessionId((current) => {
          if (current) return current;
          const next = response.sessions[0]?.session_id || newSessionId();
          localStorage.setItem(SESSION_ID_STORAGE_KEY, next);
          return next;
        });
        setServiceStatus("ready");
      })
      .catch((reason: unknown) => {
        if (cancelled) return;
        setServiceStatus("offline");
        if (reason instanceof ApiError && reason.status === 401) {
          setError("API Key 不正确，请在右上角重新填写。");
        } else {
          setError(reason instanceof Error ? reason.message : "无法读取会话列表");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [historyVersion]);

  useEffect(() => {
    if (!sessionId) {
      setHistoryLoading(false);
      return;
    }
    let cancelled = false;
    setHistoryLoading(true);
    getHistory(sessionId, apiKey)
      .then((history) => {
        if (cancelled) return;
        const loaded = history.messages.map((message) => ({
          ...message,
          id: crypto.randomUUID()
        }));
        setMessages(loaded.length ? loaded : [WELCOME]);
        setPending(history.pending_confirmation);
        setServiceStatus("ready");
        setError("");
      })
      .catch((reason: unknown) => {
        if (cancelled) return;
        setServiceStatus("offline");
        if (reason instanceof ApiError && reason.status === 401) {
          setError("API Key 不正确，请在右上角重新填写。");
        } else {
          setError(reason instanceof Error ? reason.message : "无法连接后端服务");
        }
      })
      .finally(() => {
        if (!cancelled) setHistoryLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, historyVersion]);

  async function refreshSessionList() {
    try {
      const response = await listSessions(apiKey);
      setSessions(response.sessions);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "刷新会话列表失败");
    }
  }

  async function acceptResponse(response: ChatResponse) {
    const messageId = crypto.randomUUID();
    const answer = response.answer || "（没有返回内容）";
    const chunkSize = Math.max(1, Math.ceil(answer.length / 140));
    const assistantMessage: ChatMessage = {
      id: messageId,
      role: "assistant",
      content: "",
      timestamp: new Date().toISOString(),
      trace: response.trace,
      elapsedMs: response.elapsed_ms
    };

    setMessages((current) => [...current, assistantMessage]);
    setPending(response.pending_confirmation);
    setServiceStatus("ready");

    for (let end = chunkSize; end < answer.length; end += chunkSize) {
      await wait(16);
      setMessages((current) => current.map((message) =>
        message.id === messageId
          ? { ...message, content: answer.slice(0, end) }
          : message
      ));
    }
    setMessages((current) => current.map((message) =>
      message.id === messageId ? { ...message, content: answer } : message
    ));
  }

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const content = input.trim();
    if (!content || busy) return;
    setInput("");
    setError("");
    setBusy(true);
    const assistantId = crypto.randomUUID();
    const streamingMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      timestamp: new Date().toISOString(),
      trace: [],
      streamStatus: "正在连接 Agent"
    };
    setMessages((current) => [
      ...current,
      toMessage("user", content),
      streamingMessage
    ]);

    const updateStreamingMessage = (
      updater: (message: ChatMessage) => ChatMessage
    ) => {
      setMessages((current) => current.map((message) =>
        message.id === assistantId ? updater(message) : message
      ));
    };

    let finished = false;
    try {
      for await (const event of streamChat(sessionId, content, apiKey)) {
        if (event.type === "turn_started") {
          updateStreamingMessage((message) => ({
            ...message,
            streamStatus: "正在准备上下文与记忆"
          }));
        } else if (event.type === "model_started") {
          updateStreamingMessage((message) => ({
            ...message,
            streamStatus: `正在进行第 ${event.iteration || 1} 轮模型调用`
          }));
        } else if (event.type === "answer_started") {
          updateStreamingMessage((message) => ({
            ...message,
            streamStatus: "正在生成回答"
          }));
        } else if (event.type === "answer_delta" && event.delta) {
          updateStreamingMessage((message) => ({
            ...message,
            content: message.content + event.delta,
            streamStatus: "正在生成回答"
          }));
        } else if (event.type === "answer_reset") {
          updateStreamingMessage((message) => ({
            ...message,
            content: "",
            streamStatus: "模型决定调用工具"
          }));
        } else if (event.type === "tool_started") {
          const runningTrace: ToolTraceItem = {
            call_id: event.call_id || crypto.randomUUID(),
            name: event.name || "unknown",
            arguments: event.arguments || {},
            result: "",
            status: "running",
            duration_ms: 0,
            operation_id: ""
          };
          updateStreamingMessage((message) => ({
            ...message,
            trace: [...message.trace, runningTrace],
            streamStatus: `正在执行：${toolLabel(runningTrace.name)}`
          }));
        } else if (event.type === "tool_finished") {
          const completedTrace: ToolTraceItem = {
            call_id: event.call_id || "",
            name: event.name || "unknown",
            arguments: event.arguments || {},
            result: event.result || "",
            status: event.status || "succeeded",
            duration_ms: event.duration_ms || 0,
            operation_id: event.operation_id || ""
          };
          updateStreamingMessage((message) => ({
            ...message,
            trace: message.trace.map((item) =>
              item.call_id === completedTrace.call_id ? completedTrace : item
            ),
            streamStatus: `${toolLabel(completedTrace.name)}：${statusLabel(completedTrace.status)}`
          }));
        } else if (event.type === "turn_finished" && event.response) {
          const response = event.response;
          finished = true;
          setPending(response.pending_confirmation);
          setServiceStatus("ready");
          updateStreamingMessage((message) => ({
            ...message,
            content: response.answer || message.content || "（没有返回内容）",
            trace: response.trace,
            elapsedMs: response.elapsed_ms,
            streamStatus: undefined
          }));
          void refreshSessionList();
        } else if (event.type === "error") {
          throw new Error(event.message || "Agent流式响应失败");
        }
      }
      if (!finished) throw new Error("流式连接提前结束，未收到最终结果");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "发送失败，请稍后重试");
      setServiceStatus("offline");
      updateStreamingMessage((message) => ({
        ...message,
        content: message.content || "本次请求没有完成，请稍后重试。",
        streamStatus: undefined
      }));
    } finally {
      setBusy(false);
    }
  }

  async function handleDecision(decision: "confirm" | "cancel") {
    if (!pending || busy) return;
    const active = pending;
    setBusy(true);
    setError("");
    setMessages((current) => [
      ...current,
      toMessage("user", decision === "confirm" ? "确认删除" : "取消删除")
    ]);
    try {
      const response = await decideOperation(
        active.operation_id,
        decision,
        sessionId,
        apiKey
      );
      await acceptResponse(response);
      void refreshSessionList();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "操作确认失败");
    } finally {
      setBusy(false);
    }
  }

  async function resetSession() {
    if (busy || historyLoading) return;
    const hasConversation = messages.some((message) => message.id !== WELCOME.id);
    if (!hasConversation) return;

    setBusy(true);
    setError("");
    try {
      const response = await createSession(apiKey);
      const next = response.session.session_id;
      localStorage.setItem(SESSION_ID_STORAGE_KEY, next);
      setSessions((current) => [
        response.session,
        ...current.filter((item) => item.session_id !== next)
      ]);
      setSessionId(next);
      setMessages([WELCOME]);
      setPending(null);
      setInput("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "新建会话失败");
    } finally {
      setBusy(false);
    }
  }

  function selectSession(next: string) {
    if (busy || next === sessionId) return;
    localStorage.setItem(SESSION_ID_STORAGE_KEY, next);
    setSessionId(next);
    setMessages([WELCOME]);
    setPending(null);
    setInput("");
    setError("");
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  }

  const sidebarSessions: SessionSummary[] = sessionId && !sessions.some(
    (session) => session.session_id === sessionId
  )
    ? [{
        session_id: sessionId,
        title: "新会话",
        preview: "",
        created_at: "",
        updated_at: "",
        message_count: 0
      }, ...sessions]
    : sessions;

  const visibleMessageCount = messages.filter(
    (message) => message.id !== WELCOME.id
  ).length;

  return (
    <div className="app-shell">
      <main className="workspace">
        <aside className="session-rail">
          <div className="brand">
            <img className="brand-mark" src={AGENT_AVATAR_SRC} alt="TraceMind" />
            <div>
              <strong>TraceMind</strong>
              <span>Agent Workspace</span>
            </div>
          </div>
          <button
            className="new-session"
            onClick={() => void resetSession()}
            disabled={busy || historyLoading || visibleMessageCount === 0}
          >
            <span>＋</span> 新建对话
          </button>
          <div className="rail-section">
            <span className="eyebrow">最近会话</span>
            <div className="session-list">
              {sidebarSessions.map((session) => (
                <button
                  className={`session-card ${session.session_id === sessionId ? "active" : ""}`}
                  key={session.session_id}
                  onClick={() => selectSession(session.session_id)}
                  disabled={busy}
                  title={session.preview || session.session_id}
                >
                  <div className="session-icon">⌁</div>
                  <div className="session-copy">
                    <strong>{session.title}</strong>
                    <small>{session.preview || session.session_id}</small>
                  </div>
                  <time>{sessionTimeLabel(session.updated_at)}</time>
                </button>
              ))}
            </div>
          </div>
          <div className="rail-settings">
            <label className="api-key-field">
              <span>API Key</span>
              <input
                type="password"
                value={apiKey}
                onChange={(event) => setApiKey(event.target.value)}
                onBlur={() => setHistoryVersion((value) => value + 1)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") event.currentTarget.blur();
                }}
                placeholder="可留空"
                autoComplete="off"
              />
            </label>
            <div className={`service-pill ${serviceStatus}`}>
              <i />
              {serviceStatus === "ready" ? "服务就绪" : serviceStatus === "checking" ? "检查中" : "连接异常"}
            </div>
          </div>
        </aside>

        <section className="chat-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">AGENT CONVERSATION</span>
              <h1>今天想推进什么？</h1>
            </div>
            <div className="heading-meta">
              <span className="message-count">{visibleMessageCount} 条消息</span>
              <div className={`service-pill mobile-status ${serviceStatus}`}>
                <i />
                {serviceStatus === "ready" ? "就绪" : serviceStatus === "checking" ? "检查中" : "异常"}
              </div>
            </div>
          </div>

          <div className="message-list" aria-live="polite" ref={messageListRef}>
            {historyLoading && <div className="loading-history">正在恢复会话…</div>}
            {messages.map((message) => (
              <article className={`message ${message.role}`} key={message.id}>
                <div className="avatar">
                  {message.role === "user"
                    ? "你"
                    : <AgentAvatar />}
                </div>
                <div className="message-body">
                  <div className="message-meta">
                    <strong>{message.role === "user" ? "你" : "TraceMind"}</strong>
                    <span>{timeLabel(message.timestamp)}</span>
                    {message.elapsedMs !== undefined && <span>{message.elapsedMs} ms</span>}
                  </div>
                  {message.content
                    ? <p>{message.content}</p>
                    : (
                      <div className="thinking-row">
                        <i /><i /><i /> {message.streamStatus || "正在处理任务"}
                      </div>
                    )}
                  {message.content && message.streamStatus && (
                    <div className="stream-status"><i />{message.streamStatus}</div>
                  )}
                  {message.trace.length > 0 && (
                    <details className="message-process">
                      <summary>执行过程 · {message.trace.length} 个步骤</summary>
                      <div className="process-list">
                        {message.trace.map((item, index) => (
                          <details className={`process-step ${item.status}`} key={`${item.call_id}-${index}`}>
                            <summary>
                              <span>{String(index + 1).padStart(2, "0")}</span>
                              <strong>{toolLabel(item.name)}</strong>
                              <small>{statusLabel(item.status)} · {item.duration_ms} ms</small>
                            </summary>
                            <div>
                              <label>输入参数</label>
                              <pre>{JSON.stringify(item.arguments, null, 2)}</pre>
                              <label>执行结果</label>
                              <p>{shortResult(item.result) || "无返回内容"}</p>
                            </div>
                          </details>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              </article>
            ))}
          </div>

          <div className="composer-wrap">
            {error && <div className="error-banner">{error}</div>}
            <form className="composer" onSubmit={submit}>
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder="让 Agent 阅读代码、修改文件或搜索研究资料…"
                rows={1}
                disabled={busy}
              />
              <div className="composer-footer">
                <span>Enter 发送 · Shift + Enter 换行</span>
                <button type="submit" disabled={busy || !input.trim()} aria-label="发送消息">
                  {busy ? "处理中" : "发送"} <b>↗</b>
                </button>
              </div>
            </form>
          </div>
        </section>

      </main>

      {pending && (
        <div className="modal-backdrop" role="presentation">
          <section className="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
            <div className="warning-icon">!</div>
            <span className="eyebrow danger">DESTRUCTIVE OPERATION</span>
            <h2 id="confirm-title">确认删除文件？</h2>
            <p>Agent 已暂停执行。只有你明确确认后，文件才会被移动到项目回收目录。</p>
            <dl>
              <div><dt>目标路径</dt><dd>{String(pending.arguments.path || "未知路径")}</dd></div>
              <div><dt>工具</dt><dd>{pending.tool_name}</dd></div>
              <div><dt>操作编号</dt><dd><code>{pending.operation_id}</code></dd></div>
              {pending.expires_at && <div><dt>确认有效期</dt><dd>{new Date(pending.expires_at).toLocaleString("zh-CN")}</dd></div>}
            </dl>
            <div className="modal-actions">
              <button className="cancel-button" onClick={() => void handleDecision("cancel")} disabled={busy}>取消操作</button>
              <button className="danger-button" onClick={() => void handleDecision("confirm")} disabled={busy}>{busy ? "正在执行…" : "确认删除"}</button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
