import { isValidElement, ReactNode, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import { Check, Copy, Eye } from "lucide-react";

const PREVIEWABLE_LANGS = new Set(["html", "htm", "svg"]);

function extractText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (isValidElement<{ children?: ReactNode }>(node)) {
    return extractText(node.props.children);
  }
  return "";
}

function CodeBlock({
  className,
  children,
  onPreview
}: {
  className?: string;
  children?: ReactNode;
  onPreview?: (lang: string, code: string) => void;
}) {
  const [copied, setCopied] = useState(false);
  const match = /language-([\w-]+)/.exec(className || "");
  const lang = match?.[1] || "code";
  const rawCode = extractText(children).replace(/\n$/, "");
  const canPreview = Boolean(onPreview && PREVIEWABLE_LANGS.has(lang.toLowerCase()));

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(rawCode);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard may be unavailable in non-HTTPS contexts.
    }
  }

  return (
    <div className="code-block">
      <div className="code-block-header">
        <span>{lang}</span>
        <div className="code-block-actions">
          {canPreview && (
            <button
              type="button"
              className="code-copy-btn code-preview-btn"
              onClick={() => onPreview?.(lang.toLowerCase(), rawCode)}
              aria-label="预览渲染效果"
            >
              <Eye size={13} /> 预览
            </button>
          )}
          <button
            type="button"
            className="code-copy-btn"
            onClick={() => void handleCopy()}
            aria-label="复制代码"
          >
            {copied ? (
              <>
                <Check size={13} /> 已复制
              </>
            ) : (
              <>
                <Copy size={13} /> 复制代码
              </>
            )}
          </button>
        </div>
      </div>
      <pre>
        <code className={className}>{children}</code>
      </pre>
    </div>
  );
}

export function MarkdownRenderer({
  content,
  onPreview
}: {
  content: string;
  onPreview?: (lang: string, code: string) => void;
}) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={{
          pre: ({ children }) => <>{children}</>,
          code({ className, children, ...props }) {
            const text = extractText(children);
            const isBlock = Boolean(className?.includes("language-")) || text.includes("\n");
            if (!isBlock) {
              return (
                <code className="inline-code" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <CodeBlock className={className} onPreview={onPreview}>
                {children}
              </CodeBlock>
            );
          },
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="table-scroll">
              <table>{children}</table>
            </div>
          )
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
