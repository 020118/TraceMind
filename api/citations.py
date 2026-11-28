"""Extract structured knowledge & web citations from tool execution traces."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

_UNTRUSTED_RE = re.compile(
    r"^\s*<untrusted_tool_output[^>]*>\s*([\s\S]*?)\s*</untrusted_tool_output>\s*$",
    re.IGNORECASE,
)
_WEB_ITEM_RE = re.compile(
    r"(?:^|\n)\d+\.\s+([^\n]+)\n\s*链接:\s*(https?://[^\s\n]+)(?:\n\s*摘要:\s*([^\n]+))?"
)


def _unwrap_untrusted(text: str) -> str:
    match = _UNTRUSTED_RE.match(text or "")
    return match.group(1) if match else (text or "")


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
        return host.removeprefix("www.")
    except Exception:
        return ""


def extract_citations(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse deduplicated structured citations from a flattened tool trace list."""
    citations: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for item in trace or []:
        name = str(item.get("name", ""))
        result = _unwrap_untrusted(str(item.get("result", "")))
        arguments = item.get("arguments") or {}

        if name == "knowledge_search":
            try:
                payload = json.loads(result)
            except (ValueError, TypeError):
                continue
            for hit in payload.get("results") or []:
                citation_label = str(hit.get("citation") or hit.get("source_path") or "").strip()
                if not citation_label:
                    continue
                dedupe_key = f"kb:{citation_label}"
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                citations.append(
                    {
                        "id": dedupe_key,
                        "kind": "knowledge",
                        "title": citation_label,
                        "subtitle": str(hit.get("heading") or hit.get("source_path") or ""),
                        "url": "",
                        "snippet": str(hit.get("text") or "")[:360],
                        "score": round(float(hit.get("score") or 0.0), 3),
                    }
                )

        elif name == "web_search":
            for title, url, snippet in _WEB_ITEM_RE.findall(result):
                clean_url = url.strip()
                dedupe_key = f"web:{clean_url}"
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                citations.append(
                    {
                        "id": dedupe_key,
                        "kind": "web",
                        "title": title.strip() or _domain_of(clean_url) or clean_url,
                        "subtitle": _domain_of(clean_url),
                        "url": clean_url,
                        "snippet": (snippet or "").strip()[:300],
                        "score": None,
                    }
                )

        elif name == "web_fetch":
            url = str(arguments.get("url", "")).strip()
            if not url.startswith(("http://", "https://")):
                continue
            if result.startswith("抓取失败"):
                continue
            dedupe_key = f"web:{url}"
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            citations.append(
                {
                    "id": dedupe_key,
                    "kind": "web",
                    "title": _domain_of(url) or url,
                    "subtitle": url,
                    "url": url,
                    "snippet": result[:240].strip(),
                    "score": None,
                }
            )

    return citations[:12]
