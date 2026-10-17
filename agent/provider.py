"""LLM Provider 适配层 —— 把 myAgent 的 LLMClient 包装成 proactive_v2 期望的 provider 接口。

proactive_v2 对 provider 的最小需求：
    chat(messages=, tools=, model=, max_tokens=, tool_choice=, disable_thinking=)
        -> 带 .content / .tool_calls([.id/.name/.arguments]) /
           .cache_prompt_tokens / .cache_hit_tokens 的响应对象
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class ContentSafetyError(RuntimeError):
    """内容安全拦截（与上游 provider 契约保持兼容的占位实现）"""


class ContextLengthError(RuntimeError):
    """上下文超长（与上游 provider 契约保持兼容的占位实现）"""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    cache_prompt_tokens: int | None = None
    cache_hit_tokens: int | None = None


class LLMProvider:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict = "auto",
        disable_thinking: bool = False,
        extra_body: dict | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        **_: Any,
    ) -> LLMResponse:
        resp = await self._client.chat(
            messages=messages,
            tools=list(tools) if tools else None,
            tool_choice=tool_choice,
            model=model,
            max_tokens=max_tokens,
            extra_body=extra_body,
            on_content_delta=on_content_delta,
        )
        calls: list[ToolCall] = []
        for tc in getattr(resp, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            raw_args = getattr(fn, "arguments", None) or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                args = {"_raw": str(raw_args)}
            calls.append(
                ToolCall(
                    id=str(getattr(tc, "id", "") or ""),
                    name=str(getattr(fn, "name", "") or ""),
                    arguments=args,
                )
            )
        return LLMResponse(content=getattr(resp, "content", None), tool_calls=calls)
