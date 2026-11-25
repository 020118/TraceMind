import {
  ClipboardEvent,
  DragEvent,
  FormEvent,
  KeyboardEvent,
  useEffect,
  useRef,
  useState
} from "react";
import { ArrowUp, FileText, Paperclip, Square, X } from "lucide-react";
import { resolveAssetUrl } from "../api";
import type { ChatAttachment } from "../types";

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  mode: string;
  onModeChange: (mode: string) => void;
  attachments: ChatAttachment[];
  uploadingAttachment: boolean;
  onAddFiles: (files: FileList | File[]) => void;
  onRemoveAttachment: (path: string) => void;
  busy: boolean;
  onSubmit: () => void;
  onStop: () => void;
}

export function Composer({
  value,
  onChange,
  mode,
  onModeChange,
  attachments,
  uploadingAttachment,
  onAddFiles,
  onRemoveAttachment,
  busy,
  onSubmit,
  onStop
}: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Prevent accidental submission while confirming CJK IME candidates.
    if (event.nativeEvent.isComposing || event.keyCode === 229) return;
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!busy && (value.trim() || attachments.length > 0)) {
        onSubmit();
      }
    }
  }

  function handleFormSubmit(event: FormEvent) {
    event.preventDefault();
    if (!busy && (value.trim() || attachments.length > 0)) {
      onSubmit();
    }
  }

  function handlePaste(event: ClipboardEvent<HTMLTextAreaElement>) {
    const files: File[] = [];
    for (const item of Array.from(event.clipboardData.items)) {
      if (item.kind === "file") {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }
    if (files.length > 0) {
      event.preventDefault();
      onAddFiles(files);
    }
  }

  function handleDrop(event: DragEvent<HTMLFormElement>) {
    event.preventDefault();
    setDragging(false);
    if (event.dataTransfer.files?.length) {
      onAddFiles(event.dataTransfer.files);
    }
  }

  return (
    <form
      className={`composer ${dragging ? "dragging" : ""}`}
      onSubmit={handleFormSubmit}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      {attachments.length > 0 && (
        <div className="composer-attachments">
          {attachments.map((att) => (
            <div className="composer-att-chip" key={att.path}>
              {att.is_image && att.url ? (
                <img src={resolveAssetUrl(att.url)} alt={att.name} />
              ) : (
                <FileText size={14} />
              )}
              <span title={att.name}>{att.name}</span>
              <button
                type="button"
                onClick={() => onRemoveAttachment(att.path)}
                aria-label={`移除 ${att.name}`}
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      )}

      <textarea
        ref={textareaRef}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        onPaste={handlePaste}
        placeholder="输入消息，粘贴截图或拖拽文件到此处…"
        rows={1}
        disabled={busy}
      />

      <div className="composer-toolbar">
        <div className="composer-tools-left">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".png,.jpg,.jpeg,.webp,.gif,.txt,.md,.markdown,.json,.yaml,.yml,.toml,.py,.js,.ts,.tsx,.jsx,.csv,.pdf"
            onChange={(event) => {
              if (event.target.files?.length) {
                onAddFiles(event.target.files);
                event.currentTarget.value = "";
              }
            }}
            style={{ display: "none" }}
          />
          <button
            type="button"
            className="icon-tool-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={busy || uploadingAttachment}
            title="上传图片或文件附件"
            aria-label="上传附件"
          >
            <Paperclip size={16} />
            <span>{uploadingAttachment ? "上传中…" : "附件"}</span>
          </button>

          <select
            className="composer-mode-select"
            value={mode}
            onChange={(event) => onModeChange(event.target.value)}
            aria-label="模型模式"
          >
            <option value="fast">快速 · Fast</option>
            <option value="balanced">标准 · Balanced</option>
            <option value="deep">深度 · Deep</option>
          </select>
        </div>

        <div className="composer-tools-right">
          <span className="composer-hint">Enter 发送 · Shift+Enter 换行</span>
          {busy ? (
            <button
              type="button"
              className="composer-send-btn stop"
              onClick={onStop}
              aria-label="停止生成"
            >
              <Square size={14} /> 停止
            </button>
          ) : (
            <button
              type="submit"
              className="composer-send-btn"
              disabled={(!value.trim() && attachments.length === 0) || uploadingAttachment}
              aria-label="发送消息"
            >
              <ArrowUp size={16} />
            </button>
          )}
        </div>
      </div>
    </form>
  );
}
