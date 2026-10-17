""" LLM 客户端封装 —— OpenAI 兼容接口，支持工具调用 """

import json
import logging
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMResponse:
    """LLM 响应封装 —— 统一处理文本回复和工具调用

    属性:
        content: str | None            — 文本回复（元工具调用时为 None）
        tool_calls: list | None        — 原始工具调用对象列表（来自 API）
        has_tool_calls: bool           — 是否包含工具调用
    """

    def __init__(self, message: Any):
        self.content = message.content
        self.tool_calls = message.tool_calls

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def __repr__(self) -> str:
        if self.has_tool_calls:
            names = [tc.function.name for tc in self.tool_calls]
            return f"LLMResponse(tool_calls={names})"
        preview = (self.content or "")[:50]
        return f"LLMResponse(content={preview!r})"


class LLMClient:
    """异步 LLM 客户端，与任何 OpenAI 兼容 API 对接

    支持 OpenAI、DeepSeek、通义千问、Ollama、vLLM 等。
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = "auto",
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """发送聊天请求

        messages: 对话历史（OpenAI 格式，含 system/user/assistant/tool）
        tools:    已注册工具的 schema 列表（None 表示不使用工具）
        tool_choice: 工具选择策略 ("auto" | "none" | "required" | {...})

        返回 LLMResponse，可能是文本回复或工具调用。
        """
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
        }
        if max_tokens is not None and max_tokens > 0:
            kwargs["max_tokens"] = max_tokens
        if extra_body:
            kwargs["extra_body"] = dict(extra_body)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        if on_content_delta is None:
            response = await self.client.chat.completions.create(**kwargs)
            return LLMResponse(response.choices[0].message)

        # OpenAI-compatible providers stream tool calls in fragments as well as
        # text.  Reassemble both so the normal Agent loop still receives the
        # same LLMResponse contract while text deltas can be forwarded live.
        stream = await self.client.chat.completions.create(**kwargs, stream=True)
        content_parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = {}

        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            content = getattr(delta, "content", None)
            if content:
                text = str(content)
                content_parts.append(text)
                await on_content_delta(text)

            for tool_delta in getattr(delta, "tool_calls", None) or []:
                index = int(getattr(tool_delta, "index", 0) or 0)
                state = tool_parts.setdefault(
                    index, {"id": "", "name": "", "arguments": ""}
                )
                call_id = getattr(tool_delta, "id", None)
                if call_id:
                    state["id"] += str(call_id)
                function = getattr(tool_delta, "function", None)
                if function is None:
                    continue
                name = getattr(function, "name", None)
                arguments = getattr(function, "arguments", None)
                if name:
                    state["name"] += str(name)
                if arguments:
                    state["arguments"] += str(arguments)

        tool_calls = [
            SimpleNamespace(
                id=state["id"],
                function=SimpleNamespace(
                    name=state["name"],
                    arguments=state["arguments"] or "{}",
                ),
            )
            for _, state in sorted(tool_parts.items())
        ]
        message = SimpleNamespace(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
        )
        return LLMResponse(message)
