"""轻量网页抓取工具（迁移自 akashic agent.tools.web_fetch，简化为 httpx + 标准库实现）

契约保持一致：execute(url=..., format="text") -> JSON 字符串 {"text": ...} 或 {"error": ...}
"""

from __future__ import annotations

import json
import logging
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx

from agent.host_tools.base import Tool

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 20.0
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_SKIP_TAGS = {
    "script", "style", "noscript", "template", "svg", "head",
    "nav", "footer", "aside", "form", "iframe",
}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "br", "li", "tr", "h1", "h2",
    "h3", "h4", "h5", "h6", "blockquote", "pre", "table", "ul", "ol",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [line.strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def _html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception:
        pass
    return extractor.text()


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "抓取指定 URL 页面并提取正文纯文本"
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的 URL"},
            "format": {
                "type": "string",
                "enum": ["text"],
                "description": "输出格式，当前仅支持 text",
            },
            "timeout": {
                "type": "number",
                "description": "超时秒数，默认 20",
            },
        },
        "required": ["url"],
    }

    def __init__(self, requester: Any = None) -> None:
        self._requester = requester

    async def execute(self, **kwargs: Any) -> str:
        url = str(kwargs.get("url") or "").strip()
        timeout = float(kwargs.get("timeout") or _DEFAULT_TIMEOUT)
        if not url:
            return json.dumps({"error": "url is required"}, ensure_ascii=False)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return json.dumps({"error": f"unsupported scheme: {parsed.scheme}"}, ensure_ascii=False)
        try:
            if self._requester is not None:
                resp = await self._requester.get(url)
                html = getattr(resp, "text", "") or ""
                status = getattr(resp, "status_code", 0)
                if status >= 400:
                    return json.dumps({"error": f"HTTP {status}"}, ensure_ascii=False)
            else:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=timeout,
                    headers={"User-Agent": _USER_AGENT},
                ) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    html = resp.text
        except Exception as exc:
            logger.warning("[web_fetch] %s 抓取失败: %s", url, exc)
            return json.dumps({"error": str(exc)[:200]}, ensure_ascii=False)
        return json.dumps({"url": url, "text": _html_to_text(html)}, ensure_ascii=False)
