""" 互联网搜索工具 —— 让 Agent 能搜索实时信息

使用方式：
    from agent.tools.web_search import WebSearch, WebFetch

依赖：
    pip install ddgs httpx
"""

from __future__ import annotations

import logging

import httpx
from ddgs import DDGS

from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


class WebSearch(BaseTool):
    """搜索引擎：通过 DuckDuckGo 搜索互联网（无需 API Key）"""

    name = "web_search"
    description = "搜索互联网获取实时信息，如新闻、百科、歌词、最新资讯等。当你不确定某个信息时，应该用这个工具搜索确认。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，如：丁世光 神探 歌词、2024年诺贝尔奖得主",
            },
            "max_results": {
                "type": "integer",
                "description": "返回结果数量，默认5，最多10",
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, max_results: int = 5) -> str:
        """执行互联网搜索"""
        logger.info("互联网搜索: %s", query)
        max_results = min(max(1, max_results), 10)

        try:
            results = await _search_ddg(query, max_results)
        except Exception as e:
            logger.exception("搜索失败")
            return f"搜索失败: {e}"

        if not results:
            return f"未找到 '{query}' 的相关结果"

        lines = [f"搜索结果: {query}", "=" * 40]
        for idx, r in enumerate(results, 1):
            title = r.get("title", "")
            url = r.get("href", "")
            snippet = r.get("body", "")
            lines.append(f"\n{idx}. {title}")
            lines.append(f"   链接: {url}")
            if snippet:
                lines.append(f"   摘要: {snippet[:200]}")

        return "\n".join(lines)


class WebFetch(BaseTool):
    """网页内容抓取：读取指定 URL 的文本内容"""

    name = "web_fetch"
    description = "读取指定网页的文本内容，用于查看新闻原文、歌词全文、文章详情等"
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要读取的网页 URL",
            },
            "max_chars": {
                "type": "integer",
                "description": "最多返回字符数，默认3000",
            },
        },
        "required": ["url"],
    }

    async def execute(self, url: str, max_chars: int = 3000) -> str:
        """抓取网页文本内容"""
        logger.info("网页抓取: %s", url)
        max_chars = min(max(200, max_chars), 10000)

        try:
            async with httpx.AsyncClient(
                timeout=15,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                content = resp.text
        except httpx.TimeoutException:
            return f"请求超时: {url}"
        except httpx.HTTPStatusError as e:
            return f"HTTP 错误 {e.response.status_code}: {url}"
        except Exception as e:
            return f"抓取失败: {e}"

        # 简单提取文本：去掉 HTML 标签
        text = _strip_html(content)
        text = _clean_text(text)

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n...（内容已截断）"

        return (
            f"网页内容: {url}\n"
            f"{'=' * 40}\n"
            f"{text}"
        )


# ── 辅助函数 ──────────────────────────────────────────────

async def _search_ddg(query: str, max_results: int = 5) -> list[dict]:
    """在异步线程中执行 DuckDuckGo 搜索"""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: list(DDGS().text(query, max_results=max_results)),
    )


def _clean_html(html: str) -> str:
    """去掉 HTML 标签并清理空白"""
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_html(html: str) -> str:
    """去掉 HTML 标签"""
    import re
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_text(text: str) -> str:
    """清理文本：去空行、去多余空白"""
    import re
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned.append(line)
    # 去重连续重复行
    result = []
    for line in cleaned:
        if not result or line != result[-1]:
            result.append(line)
    return "\n".join(result[:200])  # 最多200行
