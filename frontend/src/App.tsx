import { useEffect, useRef, useState } from "react";
import { Download, Menu, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import {
  ApiError,
  createSession,
  decideOperation,
  deleteKnowledge,
  deleteSession,
  getHistory,
  getUsage,
  interruptSession,
  listKnowledge,
  listSessions,
  patchSession,
  resumeTask,
  searchSessions,
  streamChat,
  truncateSession,
  uploadChatAttachment,
  uploadKnowledge
} from "./api";
import { Composer } from "./components/Composer";
import { KnowledgeModal } from "./components/KnowledgeModal";
import { CanvasDrawer, type CanvasArtifact } from "./components/CanvasDrawer";
import { MessageItem } from "./components/MessageItem";
import { SettingsModal } from "./components/SettingsModal";
import { Sidebar } from "./components/Sidebar";
import { statusLabel, toolLabel } from "./components/ThinkingBlock";
import { WelcomeView } from "./components/WelcomeView";
import { useDebouncedValue } from "./hooks/useDebouncedValue";
import { useSmartAutoScroll } from "./hooks/useSmartAutoScroll";
import type {
  ChatAttachment,
  ChatMessage,
  ChatResponse,
  KnowledgeDocument,
  PendingConfirmation,
  SearchResult,
  SessionSummary,
  ToolTraceItem,
  UsageSummary
} from "./types";

const AGENT_AVATAR_SRC = "/tracemind-avatar.jpg";
const SESSION_ID_STORAGE_KEY = "tracemind.session-id";
const API_KEY_STORAGE_KEY = "tracemind.api-key";
const THEME_STORAGE_KEY = "tracemind.theme";

function newSessionId(): string {
  return `web-${crypto.randomUUID()}`;
}

function sessionIdFromUrl(): string {
  const match = window.location.pathname.match(/^\/c\/([^/]+)/);
  return match ? decodeURIComponent(match[1]) : "";
}

function initialSessionId(): string {
  return (
    sessionIdFromUrl() ||
    localStorage.getItem(SESSION_ID_STORAGE_KEY) ||
    localStorage.getItem("myagent.session-id") ||
    ""
  );
}

function syncUrlForSession(id: string) {
  if (!id) return;
  const target = `/c/${encodeURIComponent(id)}`;
  if (window.location.pathname !== target) {
    window.history.pushState({ sessionId: id }, "", target);
  }
}

function toMessage(
  role: "user" | "assistant",
  content: string,
  trace: ToolTraceItem[] = [],
  extras: Partial<ChatMessage> = {}
): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    timestamp: new Date().toISOString(),
    trace,
    ...extras
  };
}

export default function App() {
  const [sessionId, setSessionId] = useState(initialSessionId);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [apiKey, setApiKey] = useState(
    () =>
      sessionStorage.getItem(API_KEY_STORAGE_KEY) ||
      sessionStorage.getItem("myagent.api-key") ||
      ""
  );
  const [theme, setTheme] = useState<"light" | "dark">(() =>
    localStorage.getItem(THEME_STORAGE_KEY) === "dark" ? "dark" : "light"
  );
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 850);
  const [historyVersion, setHistoryVersion] = useState(0);
  const [messagesBySession, setMessagesBySession] = useState<Record<string, ChatMessage[]>>({});
  const [busyBySession, setBusyBySession] = useState<Record<string, boolean>>({});
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [uploadingAttachment, setUploadingAttachment] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [pending, setPending] = useState<PendingConfirmation | null>(null);
  const [error, setError] = useState("");
  const [serviceStatus, setServiceStatus] = useState<"checking" | "ready" | "offline">("checking");
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [mode, setMode] = useState(() => localStorage.getItem("tracemind.mode") || "balanced");
  const [knowledge, setKnowledge] = useState<KnowledgeDocument[]>([]);
  const [knowledgeModalOpen, setKnowledgeModalOpen] = useState(false);
  const [settingsModalOpen, setSettingsModalOpen] = useState(false);
  const [canvasArtifact, setCanvasArtifact] = useState<CanvasArtifact | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const debouncedSearch = useDebouncedValue(searchQuery, 250);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [knowledgeBusy, setKnowledgeBusy] = useState(false);
  const [lastTaskTrace, setLastTaskTrace] = useState("");

  const abortControllersRef = useRef<Map<string, AbortController>>(new Map());
  const messageListRef = useRef<HTMLDivElement>(null);

  const messages = messagesBySession[sessionId] || [];
  const busy = Boolean(busyBySession[sessionId]);

  useSmartAutoScroll(messageListRef, [messages, busy, sessionId]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    sessionStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
  }, [apiKey]);

  useEffect(() => {
    const onPopState = () => {
      const fromUrl = sessionIdFromUrl();
      if (fromUrl) {
        setSessionId(fromUrl);
        localStorage.setItem(SESSION_ID_STORAGE_KEY, fromUrl);
      }
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    let cancelled = false;
    listSessions(apiKey)
      .then((response) => {
        if (cancelled) return;
        setSessions(response.sessions);
        setSessionId((current) => {
          const next = current || response.sessions[0]?.session_id || newSessionId();
          localStorage.setItem(SESSION_ID_STORAGE_KEY, next);
          syncUrlForSession(next);
          return next;
        });
        setServiceStatus("ready");
        void refreshUsage();
        void refreshKnowledge();
      })
      .catch((reason: unknown) => {
        if (cancelled) return;
        setServiceStatus("offline");
        if (reason instanceof ApiError && reason.status === 401) {
          setError("API Key 不正确，请在“设置与用量”中重新填写。");
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
    syncUrlForSession(sessionId);
    // Keep live streaming buffer intact when switching back to an in-flight session.
    if (busyBySession[sessionId]) {
      setHistoryLoading(false);
      return;
    }
    let cancelled = false;
    setHistoryLoading(true);
    getHistory(sessionId, apiKey)
      .then((history) => {
        if (cancelled) return;
        const loaded: ChatMessage[] = history.messages.map((message) => ({
          ...message,
          id: `${sessionId}-${message.seq ?? crypto.randomUUID()}`
        }));
        setMessagesBySession((prev) => ({ ...prev, [sessionId]: loaded }));
        setPending(history.pending_confirmation);
        setServiceStatus("ready");
        setError("");
      })
      .catch((reason: unknown) => {
        if (cancelled) return;
        setServiceStatus("offline");
        if (reason instanceof ApiError && reason.status === 401) {
          setError("API Key 不正确，请在“设置与用量”中重新填写。");
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

  useEffect(() => {
    const clean = debouncedSearch.trim();
    if (!clean) {
      setSearchResults([]);
      return;
    }
    let cancelled = false;
    searchSessions(clean, apiKey)
      .then((res) => {
        if (!cancelled) setSearchResults(res.results);
      })
      .catch(() => {
        if (!cancelled) setSearchResults([]);
      });
    return () => {
      cancelled = true;
    };
  }, [debouncedSearch, apiKey]);

  async function refreshSessionList() {
    try {
      const response = await listSessions(apiKey);
      setSessions(response.sessions);
      void refreshUsage();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "刷新会话列表失败");
    }
  }

  async function refreshUsage() {
    try {
      setUsage(await getUsage(apiKey));
    } catch {
      // optional observability
    }
  }

  async function refreshKnowledge() {
    try {
      setKnowledge((await listKnowledge(apiKey)).documents);
    } catch {
      // RAG may be disabled
    }
  }

  function updateSessionMessages(
    targetSessionId: string,
    updater: (list: ChatMessage[]) => ChatMessage[]
  ) {
    setMessagesBySession((prev) => ({
      ...prev,
      [targetSessionId]: updater(prev[targetSessionId] || [])
    }));
  }

  async function runStreamTurn(
    targetSessionId: string,
    content: string,
    turnAttachments: ChatAttachment[]
  ) {
    if (busyBySession[targetSessionId]) return;
    setError("");
    setBusyBySession((prev) => ({ ...prev, [targetSessionId]: true }));

    const controller = new AbortController();
    abortControllersRef.current.set(targetSessionId, controller);

    const assistantId = crypto.randomUUID();
    const userMessage = toMessage("user", content, [], { media: turnAttachments });
    const streamingMessage = toMessage("assistant", "", [], {
      id: assistantId,
      reasoning_content: "",
      isThinking: false,
      streamStatus: "正在连接 Agent"
    });

    updateSessionMessages(targetSessionId, (list) => [
      ...list,
      userMessage,
      streamingMessage
    ]);

    const patchAssistant = (updater: (m: ChatMessage) => ChatMessage) => {
      updateSessionMessages(targetSessionId, (list) =>
        list.map((m) => (m.id === assistantId ? updater(m) : m))
      );
    };

    let finished = false;
    let traceId = "";
    try {
      const mediaPaths = turnAttachments.map((item) => item.path);
      for await (const event of streamChat(
        targetSessionId,
        content,
        apiKey,
        mode,
        controller.signal,
        mediaPaths
      )) {
        if (event.type === "turn_started") {
          traceId = event.trace_id || traceId;
          patchAssistant((m) => ({
            ...m,
            traceId: event.trace_id || m.traceId,
            streamStatus: "正在准备上下文与记忆"
          }));
        } else if (event.type === "model_started") {
          patchAssistant((m) => ({
            ...m,
            streamStatus: `正在进行第 ${event.iteration || 1} 轮模型推理`
          }));
        } else if (event.type === "thinking_started") {
          patchAssistant((m) => ({
            ...m,
            isThinking: true,
            streamStatus: "正在深度思考…"
          }));
        } else if (event.type === "thinking_delta" && event.delta) {
          patchAssistant((m) => ({
            ...m,
            isThinking: true,
            reasoning_content: (m.reasoning_content || "") + event.delta,
            streamStatus: "正在深度思考…"
          }));
        } else if (event.type === "answer_started") {
          patchAssistant((m) => ({
            ...m,
            isThinking: false,
            streamStatus: "正在生成回答"
          }));
        } else if (event.type === "answer_delta" && event.delta) {
          patchAssistant((m) => ({
            ...m,
            isThinking: false,
            content: m.content + event.delta,
            streamStatus: "正在生成回答"
          }));
        } else if (event.type === "answer_reset") {
          patchAssistant((m) => ({
            ...m,
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
          patchAssistant((m) => ({
            ...m,
            isThinking: true,
            trace: [...m.trace, runningTrace],
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
          patchAssistant((m) => ({
            ...m,
            trace: m.trace.map((item) =>
              item.call_id === completedTrace.call_id ? completedTrace : item
            ),
            streamStatus: `${toolLabel(completedTrace.name)}：${statusLabel(completedTrace.status)}`
          }));
        } else if (event.type === "turn_finished" && event.response) {
          const response = event.response;
          finished = true;
          if (targetSessionId === sessionId) {
            setPending(response.pending_confirmation);
          }
          setServiceStatus("ready");
          patchAssistant((m) => ({
            ...m,
            content: response.answer || m.content || "（没有返回内容）",
            reasoning_content: response.reasoning_content || m.reasoning_content || "",
            trace: response.trace,
            elapsedMs: response.elapsed_ms,
            traceId: response.trace_id,
            isThinking: false,
            streamStatus: undefined
          }));
          // Refresh persisted seq numbers silently so edit/regenerate stay accurate.
          void getHistory(targetSessionId, apiKey)
            .then((history) => {
              const synced: ChatMessage[] = history.messages.map((msg) => ({
                ...msg,
                id: `${targetSessionId}-${msg.seq ?? crypto.randomUUID()}`
              }));
              setMessagesBySession((prev) => ({ ...prev, [targetSessionId]: synced }));
            })
            .catch(() => {});
          void refreshSessionList();
          void refreshUsage();
        } else if (event.type === "error") {
          throw new Error(event.message || "Agent 流式响应失败");
        }
      }
      if (!finished) throw new Error("流式连接提前结束，未收到最终结果");
    } catch (reason) {
      if (controller.signal.aborted) {
        setError("本轮已停止；如需继续，可重新发送或点击重试。");
        setLastTaskTrace(traceId);
        // Do NOT mark serviceStatus as offline on user-initiated stop!
      } else {
        setError(reason instanceof Error ? reason.message : "发送失败，请稍后重试");
        setServiceStatus("offline");
      }
      patchAssistant((m) => ({
        ...m,
        content: m.content || "本次请求已中止，请重试。",
        isThinking: false,
        streamStatus: undefined
      }));
    } finally {
      abortControllersRef.current.delete(targetSessionId);
      setBusyBySession((prev) => ({ ...prev, [targetSessionId]: false }));
    }
  }

  async function handleSubmit() {
    const clean = input.trim();
    if ((!clean && attachments.length === 0) || busy) return;
    const messageText = clean || "请分析我上传的附件内容。";
    const turnAttachments = [...attachments];
    setInput("");
    setAttachments([]);
    await runStreamTurn(sessionId, messageText, turnAttachments);
  }

  async function handleEditResend(target: ChatMessage, nextContent: string) {
    if (busy || target.seq === undefined) return;
    try {
      await truncateSession(sessionId, target.seq, apiKey);
      updateSessionMessages(sessionId, (list) =>
        list.filter((m) => (m.seq ?? Infinity) < target.seq!)
      );
      await runStreamTurn(sessionId, nextContent, target.media || []);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "编辑重发失败");
    }
  }

  async function handleRegenerate(assistantMsg: ChatMessage) {
    if (busy) return;
    const index = messages.findIndex((m) => m.id === assistantMsg.id);
    if (index <= 0) return;
    let prevUser: ChatMessage | undefined;
    for (let i = index - 1; i >= 0; i -= 1) {
      if (messages[i].role === "user") {
        prevUser = messages[i];
        break;
      }
    }
    if (!prevUser || prevUser.seq === undefined) return;
    try {
      await truncateSession(sessionId, prevUser.seq, apiKey);
      updateSessionMessages(sessionId, (list) =>
        list.filter((m) => (m.seq ?? Infinity) < prevUser!.seq!)
      );
      await runStreamTurn(sessionId, prevUser.content, prevUser.media || []);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "重新生成失败");
    }
  }

  async function handleAddFiles(files: FileList | File[]) {
    const list = Array.from(files);
    if (!list.length || uploadingAttachment) return;
    setUploadingAttachment(true);
    setError("");
    try {
      const uploaded: ChatAttachment[] = [];
      for (const file of list) {
        const item = await uploadChatAttachment(sessionId, file, apiKey);
        uploaded.push(item);
      }
      setAttachments((prev) => [...prev, ...uploaded].slice(0, 8));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "附件上传失败");
    } finally {
      setUploadingAttachment(false);
    }
  }

  async function stopGeneration() {
    if (!busy) return;
    abortControllersRef.current.get(sessionId)?.abort();
    try {
      await interruptSession(sessionId, apiKey);
    } catch {
      // browser abort is sufficient if backend is unreachable
    }
  }

  async function retryLastTask() {
    if (!lastTaskTrace || busy) return;
    setBusyBySession((prev) => ({ ...prev, [sessionId]: true }));
    try {
      const response: ChatResponse = await resumeTask(lastTaskTrace, sessionId, apiKey, mode);
      updateSessionMessages(sessionId, (list) => [
        ...list,
        toMessage("assistant", response.answer || "（没有返回内容）", response.trace, {
          reasoning_content: response.reasoning_content || "",
          elapsedMs: response.elapsed_ms,
          traceId: response.trace_id
        })
      ]);
      setPending(response.pending_confirmation);
      setLastTaskTrace("");
      setError("");
      void refreshSessionList();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "任务重试失败");
    } finally {
      setBusyBySession((prev) => ({ ...prev, [sessionId]: false }));
    }
  }

  async function handleDecision(decision: "confirm" | "cancel") {
    if (!pending || busy) return;
    const active = pending;
    setBusyBySession((prev) => ({ ...prev, [sessionId]: true }));
    setError("");
    updateSessionMessages(sessionId, (list) => [
      ...list,
      toMessage("user", decision === "confirm" ? "确认删除" : "取消删除")
    ]);
    try {
      const response = await decideOperation(
        active.operation_id,
        decision,
        sessionId,
        apiKey
      );
      updateSessionMessages(sessionId, (list) => [
        ...list,
        toMessage("assistant", response.answer || "（没有返回内容）", response.trace, {
          elapsedMs: response.elapsed_ms,
          traceId: response.trace_id
        })
      ]);
      setPending(response.pending_confirmation);
      void refreshSessionList();
      void refreshUsage();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "操作确认失败");
    } finally {
      setBusyBySession((prev) => ({ ...prev, [sessionId]: false }));
    }
  }

  async function resetSession() {
    if (historyLoading) return;
    setError("");
    try {
      const latest = await listSessions(apiKey);
      setSessions(latest.sessions);
      const reusable = latest.sessions.find(
        (s) => s.message_count === 0 && s.session_id !== sessionId
      );
      if (reusable) {
        localStorage.setItem(SESSION_ID_STORAGE_KEY, reusable.session_id);
        setSessionId(reusable.session_id);
        setPending(null);
        setInput("");
        setAttachments([]);
        return;
      }
      if (messages.length === 0) return;

      const response = await createSession(apiKey);
      const next = response.session.session_id;
      localStorage.setItem(SESSION_ID_STORAGE_KEY, next);
      setSessions((current) => [
        response.session,
        ...current.filter((item) => item.session_id !== next)
      ]);
      setMessagesBySession((prev) => ({ ...prev, [next]: [] }));
      setSessionId(next);
      setPending(null);
      setInput("");
      setAttachments([]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "新建会话失败");
    }
  }

  function selectSession(next: string) {
    if (next === sessionId) return;
    localStorage.setItem(SESSION_ID_STORAGE_KEY, next);
    setSessionId(next);
    setPending(null);
    setError("");
    setAttachments([]);
  }

  async function handleRenameSession(session: SessionSummary, title: string) {
    try {
      const updated = await patchSession(session.session_id, { title }, apiKey);
      setSessions((current) =>
        current.map((item) => (item.session_id === updated.session_id ? updated : item))
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "重命名会话失败");
    }
  }

  async function handleTogglePin(session: SessionSummary) {
    try {
      await patchSession(session.session_id, { pinned: !session.pinned }, apiKey);
      await refreshSessionList();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "置顶操作失败");
    }
  }

  async function removeSession(session: SessionSummary) {
    if (historyLoading) return;
    if (!window.confirm(`确定删除“${session.title || "新会话"}”吗？该会话的消息也会被删除。`)) {
      return;
    }
    setError("");
    try {
      const response = await deleteSession(session.session_id, apiKey);
      const replacement = response.replacement_session;
      setSessions((current) => [
        replacement,
        ...current.filter(
          (item) =>
            item.session_id !== session.session_id &&
            item.session_id !== replacement.session_id
        )
      ]);
      if (session.session_id === sessionId) {
        localStorage.setItem(SESSION_ID_STORAGE_KEY, replacement.session_id);
        setSessionId(replacement.session_id);
        setPending(null);
        setInput("");
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "删除会话失败");
    }
  }

  async function handleKnowledgeUpload(file: File | undefined) {
    if (!file || knowledgeBusy) return;
    setKnowledgeBusy(true);
    try {
      await uploadKnowledge(file, apiKey);
      await refreshKnowledge();
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "知识库上传失败");
    } finally {
      setKnowledgeBusy(false);
    }
  }

  async function removeKnowledge(document: KnowledgeDocument) {
    if (!window.confirm(`确定从知识库移除“${document.title}”吗？`)) return;
    try {
      await deleteKnowledge(document.id, apiKey);
      setKnowledge((current) => current.filter((item) => item.id !== document.id));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "移除知识库文档失败");
    }
  }

  const sidebarSessions: SessionSummary[] =
    sessionId && !sessions.some((s) => s.session_id === sessionId)
      ? [
          {
            session_id: sessionId,
            title: "新会话",
            preview: "",
            created_at: "",
            updated_at: "",
            message_count: messages.length,
            pinned: false
          },
          ...sessions
        ]
      : sessions;

  const activeSession = sidebarSessions.find((s) => s.session_id === sessionId);
  const lastAssistantId = [...messages].reverse().find((m) => m.role === "assistant")?.id;

  function handleExportMarkdown() {
    if (messages.length === 0) return;
    const title = activeSession?.title || "TraceMind 对话记录";
    const lines = [
      `# ${title}`,
      "",
      `> 导出时间：${new Date().toLocaleString("zh-CN")}`,
      ""
    ];
    for (const msg of messages) {
      const who = msg.role === "user" ? "你" : "TraceMind";
      const ts = msg.timestamp ? ` (${new Date(msg.timestamp).toLocaleString("zh-CN")})` : "";
      lines.push(`## ${who}${ts}`, "", msg.content || "", "");
    }
    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const safeName = title.replace(/[\\/:*?"<>|]/g, "_").slice(0, 40) || "tracemind-chat";
    a.href = url;
    a.download = `${safeName}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app-shell">
      <main className={`workspace ${sidebarOpen ? "sidebar-open" : "sidebar-collapsed"}`}>
        <Sidebar
          open={sidebarOpen}
          avatarSrc={AGENT_AVATAR_SRC}
          sessions={sidebarSessions}
          activeSessionId={sessionId}
          busySessions={busyBySession}
          searchQuery={searchQuery}
          searchResults={searchResults}
          knowledgeCount={knowledge.length}
          serviceStatus={serviceStatus}
          theme={theme}
          onCloseMobile={() => {
            if (window.innerWidth <= 850) setSidebarOpen(false);
          }}
          onNewSession={() => void resetSession()}
          onSelectSession={selectSession}
          onRenameSession={(s, title) => void handleRenameSession(s, title)}
          onTogglePin={(s) => void handleTogglePin(s)}
          onDeleteSession={(s) => void removeSession(s)}
          onSearchChange={setSearchQuery}
          onOpenKnowledge={() => setKnowledgeModalOpen(true)}
          onOpenSettings={() => setSettingsModalOpen(true)}
          onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
        />

        <section className="chat-panel">
          <header className="panel-heading">
            <div className="heading-left">
              <button
                type="button"
                className="sidebar-toggle-btn"
                onClick={() => setSidebarOpen((v) => !v)}
                aria-label={sidebarOpen ? "收起侧边栏" : "展开侧边栏"}
                title={sidebarOpen ? "收起侧边栏" : "展开侧边栏"}
              >
                <span className="mobile-only"><Menu size={18} /></span>
                <span className="desktop-only">
                  {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
                </span>
              </button>
              <div className="heading-titles">
                <span className="eyebrow">TRACEMIND CONVERSATION</span>
                <h1>{activeSession?.title || "新会话"}</h1>
              </div>
            </div>
            <div className="heading-meta">
              {messages.length > 0 && (
                <button
                  type="button"
                  className="export-btn"
                  onClick={handleExportMarkdown}
                  title="导出对话为 Markdown"
                >
                  <Download size={13} />
                  <span>导出</span>
                </button>
              )}
              <span className="message-count">{messages.length} 条消息</span>
              <div className={`service-pill ${serviceStatus}`}>
                <i />
                {serviceStatus === "ready"
                  ? "就绪"
                  : serviceStatus === "checking"
                    ? "检查中"
                    : "异常"}
              </div>
            </div>
          </header>

          <div className="message-list" aria-live="polite" ref={messageListRef}>
            {historyLoading && <div className="loading-history">正在恢复会话…</div>}
            {!historyLoading && messages.length === 0 ? (
              <WelcomeView
                avatarSrc={AGENT_AVATAR_SRC}
                onSelectPrompt={(prompt) => {
                  setInput(prompt);
                }}
              />
            ) : (
              <div className="message-stream-inner">
                {messages.map((message) => (
                  <MessageItem
                    key={message.id}
                    message={message}
                    avatarSrc={AGENT_AVATAR_SRC}
                    busy={busy}
                    canRegenerate={message.id === lastAssistantId}
                    onEditResend={(m, next) => void handleEditResend(m, next)}
                    onRegenerate={(m) => void handleRegenerate(m)}
                    onPreviewArtifact={(lang, code) => setCanvasArtifact({ lang, code })}
                  />
                ))}
              </div>
            )}
          </div>

          <div className="composer-wrap">
            <div className="composer-inner">
              {error && (
                <div className="error-banner">
                  <span>{error}</span>
                  {lastTaskTrace && (
                    <button type="button" onClick={() => void retryLastTask()}>
                      重试任务
                    </button>
                  )}
                </div>
              )}
              <Composer
                value={input}
                onChange={setInput}
                mode={mode}
                onModeChange={(nextMode) => {
                  setMode(nextMode);
                  localStorage.setItem("tracemind.mode", nextMode);
                }}
                attachments={attachments}
                uploadingAttachment={uploadingAttachment}
                onAddFiles={(files) => void handleAddFiles(files)}
                onRemoveAttachment={(path) =>
                  setAttachments((prev) => prev.filter((item) => item.path !== path))
                }
                busy={busy}
                onSubmit={() => void handleSubmit()}
                onStop={() => void stopGeneration()}
              />
            </div>
          </div>
        </section>
        {canvasArtifact && (
          <CanvasDrawer
            artifact={canvasArtifact}
            onClose={() => setCanvasArtifact(null)}
          />
        )}
      </main>

      <KnowledgeModal
        open={knowledgeModalOpen}
        documents={knowledge}
        busy={knowledgeBusy}
        onClose={() => setKnowledgeModalOpen(false)}
        onUpload={(file) => void handleKnowledgeUpload(file)}
        onDelete={(doc) => void removeKnowledge(doc)}
      />

      <SettingsModal
        open={settingsModalOpen}
        apiKey={apiKey}
        serviceStatus={serviceStatus}
        usage={usage}
        theme={theme}
        onClose={() => setSettingsModalOpen(false)}
        onApiKeyChange={setApiKey}
        onApiKeyCommit={() => setHistoryVersion((v) => v + 1)}
        onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
      />

      {pending && (
        <div className="modal-backdrop" role="presentation">
          <section
            className="confirm-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="confirm-title"
          >
            <div className="warning-icon">!</div>
            <span className="eyebrow danger">DESTRUCTIVE OPERATION</span>
            <h2 id="confirm-title">确认执行高危操作？</h2>
            <p>Agent 已暂停执行。只有你明确确认后，该操作才会继续执行。</p>
            <dl>
              <div>
                <dt>目标路径</dt>
                <dd>{String(pending.arguments.path || "未知路径")}</dd>
              </div>
              <div>
                <dt>工具</dt>
                <dd>{pending.tool_name}</dd>
              </div>
              <div>
                <dt>操作编号</dt>
                <dd><code>{pending.operation_id}</code></dd>
              </div>
              {pending.expires_at && (
                <div>
                  <dt>确认有效期</dt>
                  <dd>{new Date(pending.expires_at).toLocaleString("zh-CN")}</dd>
                </div>
              )}
            </dl>
            <div className="modal-actions">
              <button
                type="button"
                className="cancel-button"
                onClick={() => void handleDecision("cancel")}
                disabled={busy}
              >
                取消操作
              </button>
              <button
                type="button"
                className="danger-button"
                onClick={() => void handleDecision("confirm")}
                disabled={busy}
              >
                {busy ? "正在执行…" : "确认执行"}
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
