import type { ToolTraceItem } from "../types";

export interface CitationItem {
  id: string;
  kind: "knowledge" | "web";
  title: string;
  subtitle?: string;
  url?: string;
  snippet?: string;
  score?: number | null;
}

function unwrapUntrusted(text: string): string {
  const match = /^\s*<untrusted_tool_output[^>]*>\s*([\s\S]*?)\s*<\/untrusted_tool_output>\s*$/i.exec(
    text || ""
  );
  return match ? match[1] : text || "";
}

function domainOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

export function extractCitations(trace: ToolTraceItem[]): CitationItem[] {
  const citations: CitationItem[] = [];
  const seen = new Set<string>();

  for (const item of trace || []) {
    const raw = unwrapUntrusted(item.result || "");

    if (item.name === "knowledge_search") {
      try {
        const parsed = JSON.parse(raw) as {
          results?: Array<{
            citation?: string;
            source_path?: string;
            heading?: string;
            text?: string;
            score?: number;
          }>;
        };
        for (const hit of parsed.results || []) {
          const title = (hit.citation || hit.source_path || "").trim();
          if (!title) continue;
          const key = `kb:${title}`;
          if (seen.has(key)) continue;
          seen.add(key);
          citations.push({
            id: key,
            kind: "knowledge",
            title,
            subtitle: hit.heading || hit.source_path || "",
            snippet: (hit.text || "").slice(0, 360),
            score: typeof hit.score === "number" ? Math.round(hit.score * 1000) / 1000 : null
          });
        }
      } catch {
        // Ignore non-JSON error messages.
      }
    } else if (item.name === "web_search") {
      const regex =
        /(?:^|\n)\d+\.\s+([^\n]+)\n\s*链接:\s*(https?:\/\/[^\s\n]+)(?:\n\s*摘要:\s*([^\n]+))?/g;
      let match: RegExpExecArray | null;
      while ((match = regex.exec(raw)) !== null) {
        const title = (match[1] || "").trim();
        const url = (match[2] || "").trim();
        const snippet = (match[3] || "").trim();
        const key = `web:${url}`;
        if (seen.has(key)) continue;
        seen.add(key);
        citations.push({
          id: key,
          kind: "web",
          title: title || domainOf(url) || url,
          subtitle: domainOf(url),
          url,
          snippet: snippet.slice(0, 300)
        });
      }
    } else if (item.name === "web_fetch") {
      const url = String(item.arguments?.url || "").trim();
      if (!url.startsWith("http://") && !url.startsWith("https://")) continue;
      if (raw.startsWith("抓取失败")) continue;
      const key = `web:${url}`;
      if (seen.has(key)) continue;
      seen.add(key);
      citations.push({
        id: key,
        kind: "web",
        title: domainOf(url) || url,
        subtitle: url,
        url,
        snippet: raw.slice(0, 240).trim()
      });
    }
  }

  return citations.slice(0, 12);
}
