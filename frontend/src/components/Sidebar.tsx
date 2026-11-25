import { useState } from "react";
import {
  BookOpen,
  Check,
  MessageSquare,
  Moon,
  Pencil,
  Pin,
  Plus,
  Search,
  Settings,
  Sun,
  Trash2,
  X
} from "lucide-react";
import type { SearchResult, SessionSummary } from "../types";

function sessionGroupLabel(session: SessionSummary): string {
  if (session.pinned) return "置顶会话";
  if (!session.updated_at) return "今天";
  const parsed = new Date(session.updated_at);
  if (Number.isNaN(parsed.getTime())) return "更早";
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const diffDays = Math.floor((startOfToday - new Date(parsed.getFullYear(), parsed.getMonth(), parsed.getDate()).getTime()) / 86_400_000);
  if (diffDays <= 0) return "今天";
  if (diffDays === 1) return "昨天";
  if (diffDays <= 7) return "最近 7 天";
  return "更早";
}

interface SidebarProps {
  open: boolean;
  avatarSrc: string;
  sessions: SessionSummary[];
  activeSessionId: string;
  busySessions: Record<string, boolean>;
  searchQuery: string;
  searchResults: SearchResult[];
  knowledgeCount: number;
  serviceStatus: "checking" | "ready" | "offline";
  theme: "light" | "dark";
  onCloseMobile: () => void;
  onNewSession: () => void;
  onSelectSession: (sessionId: string) => void;
  onRenameSession: (session: SessionSummary, title: string) => void;
  onTogglePin: (session: SessionSummary) => void;
  onDeleteSession: (session: SessionSummary) => void;
  onSearchChange: (value: string) => void;
  onOpenKnowledge: () => void;
  onOpenSettings: () => void;
  onToggleTheme: () => void;
}

export function Sidebar({
  open,
  avatarSrc,
  sessions,
  activeSessionId,
  busySessions,
  searchQuery,
  searchResults,
  knowledgeCount,
  serviceStatus,
  theme,
  onCloseMobile,
  onNewSession,
  onSelectSession,
  onRenameSession,
  onTogglePin,
  onDeleteSession,
  onSearchChange,
  onOpenKnowledge,
  onOpenSettings,
  onToggleTheme
}: SidebarProps) {
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");

  const groups = new Map<string, SessionSummary[]>();
  for (const session of sessions) {
    const label = sessionGroupLabel(session);
    const list = groups.get(label) || [];
    list.push(session);
    groups.set(label, list);
  }

  function startRename(session: SessionSummary) {
    setRenamingId(session.session_id);
    setRenameDraft(session.title || "新会话");
  }

  function commitRename(session: SessionSummary) {
    const clean = renameDraft.trim();
    setRenamingId(null);
    if (clean && clean !== session.title) {
      onRenameSession(session, clean);
    }
  }

  return (
    <>
      {open && <div className="sidebar-backdrop" onClick={onCloseMobile} />}
      <aside className={`session-rail ${open ? "open" : "collapsed"}`}>
        <div className="rail-top">
          <div className="brand">
            <img className="brand-mark" src={avatarSrc} alt="TraceMind" />
            <div>
              <strong>TraceMind</strong>
              <span>Agent Workspace</span>
            </div>
          </div>
          <button
            type="button"
            className="mobile-close-btn"
            onClick={onCloseMobile}
            aria-label="关闭侧边栏"
          >
            <X size={16} />
          </button>
        </div>

        <button
          type="button"
          className="new-session"
          onClick={() => {
            onNewSession();
            onCloseMobile();
          }}
        >
          <Plus size={16} /> 新建对话
        </button>

        <label className="session-search">
          <Search size={13} />
          <input
            value={searchQuery}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder="搜索所有会话…"
          />
          {searchQuery && (
            <button
              type="button"
              className="search-clear"
              onClick={() => onSearchChange("")}
              aria-label="清空搜索"
            >
              <X size={12} />
            </button>
          )}
        </label>

        {searchResults.length > 0 && (
          <div className="search-results">
            {searchResults.slice(0, 8).map((result) => (
              <button
                type="button"
                key={result.message_id}
                onClick={() => {
                  onSelectSession(result.session_id);
                  onCloseMobile();
                }}
              >
                <strong>
                  {result.content.slice(0, 52)}
                  {result.content.length > 52 ? "…" : ""}
                </strong>
                <small>
                  {result.role === "user" ? "你" : "TraceMind"} · {result.session_id.slice(0, 14)}
                </small>
              </button>
            ))}
          </div>
        )}

        <div className="rail-section">
          <div className="session-list">
            {Array.from(groups.entries()).map(([groupLabel, items]) => (
              <div className="session-group" key={groupLabel}>
                <span className="eyebrow">{groupLabel}</span>
                {items.map((session) => {
                  const isActive = session.session_id === activeSessionId;
                  const isGenerating = Boolean(busySessions[session.session_id]);
                  const isRenaming = renamingId === session.session_id;
                  return (
                    <div
                      className={`session-row ${isActive ? "active" : ""}`}
                      key={session.session_id}
                    >
                      {isRenaming ? (
                        <div className="session-rename-row">
                          <input
                            value={renameDraft}
                            onChange={(event) => setRenameDraft(event.target.value)}
                            onKeyDown={(event) => {
                              if (event.key === "Enter") commitRename(session);
                              if (event.key === "Escape") setRenamingId(null);
                            }}
                            onBlur={() => commitRename(session)}
                            autoFocus
                          />
                          <button
                            type="button"
                            onMouseDown={(event) => event.preventDefault()}
                            onClick={() => commitRename(session)}
                            aria-label="保存标题"
                          >
                            <Check size={13} />
                          </button>
                        </div>
                      ) : (
                        <>
                          <button
                            type="button"
                            className={`session-card ${isActive ? "active" : ""}`}
                            onClick={() => {
                              onSelectSession(session.session_id);
                              onCloseMobile();
                            }}
                            title={session.preview || session.title}
                          >
                            <span className={`session-icon ${isGenerating ? "busy" : ""}`}>
                              {session.pinned ? <Pin size={12} /> : <MessageSquare size={12} />}
                            </span>
                            <span className="session-copy">
                              <strong>{session.title}</strong>
                              <small>
                                {isGenerating
                                  ? "正在生成…"
                                  : session.preview || "新会话"}
                              </small>
                            </span>
                          </button>
                          <div className="session-row-actions">
                            <button
                              type="button"
                              onClick={() => onTogglePin(session)}
                              title={session.pinned ? "取消置顶" : "置顶会话"}
                              aria-label={session.pinned ? "取消置顶" : "置顶会话"}
                            >
                              <Pin size={12} />
                            </button>
                            <button
                              type="button"
                              onClick={() => startRename(session)}
                              title="重命名"
                              aria-label="重命名会话"
                            >
                              <Pencil size={12} />
                            </button>
                            <button
                              type="button"
                              className="danger"
                              onClick={() => onDeleteSession(session)}
                              title="删除会话"
                              aria-label="删除会话"
                            >
                              <Trash2 size={12} />
                            </button>
                          </div>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        <div className="rail-footer">
          <button type="button" className="rail-action-btn" onClick={onOpenKnowledge}>
            <BookOpen size={15} />
            <span>知识库</span>
            <em className="badge">{knowledgeCount}</em>
          </button>
          <button type="button" className="rail-action-btn" onClick={onOpenSettings}>
            <Settings size={15} />
            <span>设置与用量</span>
            <i className={`status-dot ${serviceStatus}`} />
          </button>
          <button
            type="button"
            className="rail-action-btn"
            onClick={onToggleTheme}
            aria-label="切换主题"
          >
            {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
            <span>{theme === "dark" ? "浅色模式" : "深色模式"}</span>
          </button>
        </div>
      </aside>
    </>
  );
}
