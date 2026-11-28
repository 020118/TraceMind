import { useState } from "react";
import { Check, Copy, FileText, Pencil, RotateCcw, X } from "lucide-react";
import { resolveAssetUrl } from "../api";
import type { ChatMessage } from "../types";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { ThinkingBlock } from "./ThinkingBlock";

function timeLabel(value: string): string {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "";
  return parsed.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit"
  });
}

interface MessageItemProps {
  message: ChatMessage;
  avatarSrc: string;
  busy: boolean;
  canRegenerate: boolean;
  onEditResend: (message: ChatMessage, nextContent: string) => void;
  onRegenerate: (message: ChatMessage) => void;
  onPreviewArtifact?: (lang: string, code: string) => void;
}

export function MessageItem({
  message,
  avatarSrc,
  busy,
  canRegenerate,
  onEditResend,
  onRegenerate,
  onPreviewArtifact
}: MessageItemProps) {
  const [copied, setCopied] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      // ignore
    }
  }

  function submitEdit() {
    const clean = draft.trim();
    if (!clean || busy) return;
    setEditing(false);
    onEditResend(message, clean);
  }

  const isStreaming = Boolean(message.streamStatus);

  return (
    <article className={`message ${message.role}`}>
      <div className="avatar">
        {message.role === "user" ? "你" : <img src={avatarSrc} alt="TraceMind" />}
      </div>
      <div className="message-body">
        <div className="message-meta">
          <strong>{message.role === "user" ? "你" : "TraceMind"}</strong>
          <span>{timeLabel(message.timestamp)}</span>
          {message.elapsedMs !== undefined && <span>{message.elapsedMs} ms</span>}
          {message.traceId && (
            <span className="trace-id" title={message.traceId}>
              {message.traceId.slice(0, 12)}…
            </span>
          )}
        </div>

        {message.role === "assistant" && (
          <ThinkingBlock
            reasoning={message.reasoning_content}
            trace={message.trace}
            isThinking={message.isThinking}
            streamStatus={message.streamStatus}
          />
        )}

        {message.media && message.media.length > 0 && (
          <div className="message-attachments">
            {message.media.map((att, idx) =>
              att.is_image && att.url ? (
                <a
                  key={`${att.path}-${idx}`}
                  href={resolveAssetUrl(att.url)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="attachment-image-link"
                >
                  <img src={resolveAssetUrl(att.url)} alt={att.name} />
                </a>
              ) : (
                <span key={`${att.path}-${idx}`} className="attachment-file-chip">
                  <FileText size={12} />
                  {att.name}
                </span>
              )
            )}
          </div>
        )}

        {editing ? (
          <div className="message-edit-box">
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              rows={3}
              autoFocus
            />
            <div className="message-edit-actions">
              <button
                type="button"
                className="ghost-btn"
                onClick={() => {
                  setEditing(false);
                  setDraft(message.content);
                }}
              >
                <X size={13} /> 取消
              </button>
              <button
                type="button"
                className="primary-btn"
                onClick={submitEdit}
                disabled={!draft.trim() || busy}
              >
                保存并重新发送
              </button>
            </div>
          </div>
        ) : message.role === "user" ? (
          <div className="message-bubble user-bubble">{message.content}</div>
        ) : message.content ? (
          <div className="message-bubble assistant-bubble">
            <MarkdownRenderer content={message.content} onPreview={onPreviewArtifact} />
          </div>
        ) : (
          <div className="thinking-row">
            <i />
            <i />
            <i /> {message.streamStatus || "正在处理任务"}
          </div>
        )}

        {message.content && isStreaming && (
          <div className="stream-status">
            <i />
            {message.streamStatus}
          </div>
        )}

        {!editing && !isStreaming && message.content && (
          <div className="message-actions">
            <button
              type="button"
              onClick={() => void handleCopy()}
              title="复制内容"
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
              <span>{copied ? "已复制" : "复制"}</span>
            </button>
            {message.role === "user" && message.seq !== undefined && (
              <button
                type="button"
                disabled={busy}
                onClick={() => {
                  setDraft(message.content);
                  setEditing(true);
                }}
                title="编辑并重新发送"
              >
                <Pencil size={13} />
                <span>编辑</span>
              </button>
            )}
            {message.role === "assistant" && canRegenerate && (
              <button
                type="button"
                disabled={busy}
                onClick={() => onRegenerate(message)}
                title="重新生成回答"
              >
                <RotateCcw size={13} />
                <span>重新生成</span>
              </button>
            )}
          </div>
        )}
      </div>
    </article>
  );
}
