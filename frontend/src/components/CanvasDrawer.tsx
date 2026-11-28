import { useState } from "react";
import { Check, Code2, Copy, Download, Eye, X } from "lucide-react";

export interface CanvasArtifact {
  lang: string;
  code: string;
}

interface CanvasDrawerProps {
  artifact: CanvasArtifact | null;
  onClose: () => void;
}

function buildPreviewDoc(lang: string, code: string): string {
  const normalized = lang.toLowerCase();
  if (normalized === "svg") {
    return `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<style>
  html, body {
    margin: 0;
    padding: 20px;
    min-height: 100%;
    box-sizing: border-box;
    display: grid;
    place-items: center;
    background: #f8f6ef;
    font-family: system-ui, -apple-system, sans-serif;
  }
  svg {
    max-width: 100%;
    height: auto;
  }
</style>
</head>
<body>${code}</body>
</html>`;
  }
  return code;
}

export function CanvasDrawer({ artifact, onClose }: CanvasDrawerProps) {
  const [tab, setTab] = useState<"preview" | "source">("preview");
  const [copied, setCopied] = useState(false);

  if (!artifact) return null;

  const isSvg = artifact.lang.toLowerCase() === "svg";
  const srcDoc = buildPreviewDoc(artifact.lang, artifact.code);

  async function handleCopy() {
    if (!artifact) return;
    try {
      await navigator.clipboard.writeText(artifact.code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      // ignore
    }
  }

  function handleDownload() {
    if (!artifact) return;
    const ext = isSvg ? "svg" : "html";
    const mime = isSvg ? "image/svg+xml;charset=utf-8" : "text/html;charset=utf-8";
    const blob = new Blob([artifact.code], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `tracemind-artifact.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <aside className="canvas-drawer" aria-label="实时画布预览">
      <header className="canvas-header">
        <div className="canvas-title">
          <span className="eyebrow">CANVAS PREVIEW</span>
          <strong>{artifact.lang.toUpperCase()} 实时渲染</strong>
        </div>
        <div className="canvas-controls">
          <div className="canvas-tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={tab === "preview"}
              className={tab === "preview" ? "active" : ""}
              onClick={() => setTab("preview")}
            >
              <Eye size={13} /> 预览
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === "source"}
              className={tab === "source" ? "active" : ""}
              onClick={() => setTab("source")}
            >
              <Code2 size={13} /> 源码
            </button>
          </div>
          <button
            type="button"
            className="canvas-icon-btn"
            onClick={() => void handleCopy()}
            title="复制源码"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
          <button
            type="button"
            className="canvas-icon-btn"
            onClick={handleDownload}
            title="下载文件"
          >
            <Download size={14} />
          </button>
          <button
            type="button"
            className="canvas-icon-btn"
            onClick={onClose}
            aria-label="关闭预览"
            title="关闭预览"
          >
            <X size={15} />
          </button>
        </div>
      </header>
      <div className="canvas-body">
        {tab === "preview" ? (
          <iframe
            className="canvas-iframe"
            title="TraceMind Artifact Preview"
            sandbox="allow-scripts"
            srcDoc={srcDoc}
          />
        ) : (
          <pre className="canvas-source">
            <code>{artifact.code}</code>
          </pre>
        )}
      </div>
    </aside>
  );
}
