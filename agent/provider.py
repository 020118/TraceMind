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
    """内容安全拦截（由上游安全审核或内容过滤触发）"""


class ContextLengthError(RuntimeError):
    """上下文超长（超过模型最大上下文窗口）"""


def _classify_provider_error(exc: Exception) -> Exception:
    """Translate provider-specific error messages into typed domain errors."""
    if isinstance(exc, (ContentSafetyError, ContextLengthError)):
        return exc
    msg = str(exc).lower()
    if any(
        token in msg
        for token in (
            "context_length_exceeded",
            "maximum context length",
            "context window",
            "too many tokens",
            "token limit",
            "input length",
        )
    ):
        return ContextLengthError(str(exc))
    if any(
        token in msg
        for token in (
            "content_filter",
            "content_policy",
            "data_inspection_failed",
            "inappropriate content",
            "safety policy",
        )
    ):
        return ContentSafetyError(str(exc))
    return exc


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str | None = None
    cache_prompt_tokens: int | None = None
    cache_hit_tokens: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


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
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        **_: Any,
    ) -> LLMResponse:
        merged_extra = dict(extra_body or {})
        if disable_thinking:
            merged_extra.setdefault("enable_thinking", False)
        client_kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": list(tools) if tools else None,
            "tool_choice": tool_choice,
            "model": model,
            "max_tokens": max_tokens,
            "extra_body": merged_extra or None,
            "on_content_delta": on_content_delta,
        }
        if on_reasoning_delta is not None:
            client_kwargs["on_reasoning_delta"] = on_reasoning_delta
        try:
            resp = await self._client.chat(**client_kwargs)
        except Exception as exc:
            classified = _classify_provider_error(exc)
            if classified is not exc:
                raise classified from exc
            raise
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
        return LLMResponse(
            content=getattr(resp, "content", None),
            reasoning_content=getattr(resp, "reasoning_content", None),
            tool_calls=calls,
            prompt_tokens=int(getattr(resp, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(resp, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(resp, "total_tokens", 0) or 0),
        )
