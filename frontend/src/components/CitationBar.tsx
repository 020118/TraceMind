import { useMemo, useState } from "react";
import { BookOpen, ExternalLink, Globe, X } from "lucide-react";
import type { ToolTraceItem } from "../types";
import { extractCitations, type CitationItem } from "../utils/citations";

interface CitationBarProps {
  trace: ToolTraceItem[];
}

export function CitationBar({ trace }: CitationBarProps) {
  const citations = useMemo(() => extractCitations(trace), [trace]);
  const [activeKb, setActiveKb] = useState<CitationItem | null>(null);

  if (citations.length === 0) return null;

  return (
    <div className="citation-bar" aria-label="来源引用">
      <div className="citation-list">
        {citations.map((item) =>
          item.kind === "web" && item.url ? (
            <a
              key={item.id}
              href={item.url}
              target="_blank"
              rel="noopener noreferrer"
              className="citation-chip web"
              title={item.snippet || item.url}
            >
              <Globe size={12} />
              <span className="citation-title">{item.title}</span>
              {item.subtitle && <small>{item.subtitle}</small>}
              <ExternalLink size={11} />
            </a>
          ) : (
            <button
              key={item.id}
              type="button"
              className={`citation-chip knowledge ${activeKb?.id === item.id ? "active" : ""}`}
              onClick={() =>
                setActiveKb((prev) => (prev?.id === item.id ? null : item))
              }
              title={item.subtitle || item.title}
            >
              <BookOpen size={12} />
              <span className="citation-title">{item.title}</span>
              {item.score !== null && item.score !== undefined && (
                <small>{Math.round(item.score * 100)}%</small>
              )}
            </button>
          )
        )}
      </div>

      {activeKb && (
        <div className="citation-snippet-panel">
          <div className="citation-snippet-header">
            <strong>
              <BookOpen size={12} /> {activeKb.title}
            </strong>
            <button
              type="button"
              onClick={() => setActiveKb(null)}
              aria-label="关闭原文摘录"
            >
              <X size={13} />
            </button>
          </div>
          {activeKb.subtitle && (
            <div className="citation-snippet-sub">{activeKb.subtitle}</div>
          )}
          <p>{activeKb.snippet || "（无原文摘录）"}</p>
        </div>
      )}
    </div>
  );
}
