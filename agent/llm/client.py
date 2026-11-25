""" LLM 客户端封装 —— OpenAI 兼容接口，支持工具调用、超时与首字前重试 """

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """Return True only for transient rate-limit / server / network errors."""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS_CODES
    msg = str(exc).lower()
    if any(code in msg for code in ("429", "rate limit", "too many requests", "502", "503", "504", "timed out", "connection error", "temporarily unavailable")):
        return True
    return False


class LLMResponse:
    """LLM 响应封装 —— 统一处理文本回复和工具调用

    属性:
        content: str | None            — 文本回复（元工具调用时为 None）
        tool_calls: list | None        — 原始工具调用对象列表（来自 API）
        has_tool_calls: bool           — 是否包含工具调用
    """

    def __init__(self, message: Any, usage: Any = None):
        self.content = message.content
        self.reasoning_content = getattr(message, "reasoning_content", None)
        self.tool_calls = message.tool_calls
        self.usage = usage
        self.prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        self.total_tokens = int(
            getattr(usage, "total_tokens", 0)
            or self.prompt_tokens + self.completion_tokens
        )

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

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        usage_recorder: Callable[[dict[str, Any]], None] | None = None,
        usage_purpose: str = "auxiliary",
        request_timeout_seconds: float = 60.0,
        max_retries: int = 2,
        retry_base_delay_seconds: float = 0.6,
    ):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=request_timeout_seconds,
        )
        self.model = model
        self._usage_recorder = usage_recorder
        self._usage_purpose = usage_purpose
        self._request_timeout = max(1.0, float(request_timeout_seconds))
        self._max_retries = max(0, int(max_retries))
        self._retry_base_delay = max(0.0, float(retry_base_delay_seconds))

    def _record_usage(self, response: Any, *, model: str) -> None:
        if self._usage_recorder is None:
            return
        usage = getattr(response, "usage", None)
        self._usage_recorder(
            {
                "model": model,
                "purpose": self._usage_purpose,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            }
        )

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
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
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

        if on_content_delta is None and on_reasoning_delta is None:
            for attempt in range(self._max_retries + 1):
                try:
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(**kwargs),
                        timeout=self._request_timeout,
                    )
                    self._record_usage(response, model=model or self.model)
                    return LLMResponse(
                        response.choices[0].message,
                        getattr(response, "usage", None),
                    )
                except Exception as exc:
                    if attempt >= self._max_retries or not _is_retryable_llm_error(exc):
                        raise
                    delay = self._retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "LLM 非流式调用失败，将在 %.2fs 后重试 (%d/%d): %s",
                        delay,
                        attempt + 1,
                        self._max_retries,
                        exc,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)

        # OpenAI-compatible providers stream tool calls in fragments as well as
        # text.  Only retry BEFORE any token/tool delta has been emitted so we
        # never duplicate partial output to the user.
        for attempt in range(self._max_retries + 1):
            emitted_any = False
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_parts: dict[int, dict[str, str]] = {}
            usage = None
            try:
                try:
                    stream = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            **kwargs,
                            stream=True,
                            stream_options={"include_usage": True},
                        ),
                        timeout=self._request_timeout,
                    )
                except Exception as exc:
                    if "stream_options" not in str(exc).lower():
                        raise
                    stream = await asyncio.wait_for(
                        self.client.chat.completions.create(**kwargs, stream=True),
                        timeout=self._request_timeout,
                    )

                async for chunk in stream:
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage is not None:
                        usage = chunk_usage
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    if delta is None:
                        continue

                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        emitted_any = True
                        r_text = str(reasoning)
                        reasoning_parts.append(r_text)
                        if on_reasoning_delta is not None:
                            await on_reasoning_delta(r_text)

                    content = getattr(delta, "content", None)
                    if content:
                        emitted_any = True
                        text = str(content)
                        content_parts.append(text)
                        if on_content_delta is not None:
                            await on_content_delta(text)

                    for tool_delta in getattr(delta, "tool_calls", None) or []:
                        emitted_any = True
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
                break
            except Exception as exc:
                if (
                    emitted_any
                    or attempt >= self._max_retries
                    or not _is_retryable_llm_error(exc)
                ):
                    raise
                delay = self._retry_base_delay * (2 ** attempt)
                logger.warning(
                    "LLM 流式首字前失败，将在 %.2fs 后重试 (%d/%d): %s",
                    delay,
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if delay > 0:
                    await asyncio.sleep(delay)

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
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=tool_calls,
        )
        if self._usage_recorder is not None:
            self._usage_recorder(
                {
                    "model": model or self.model,
                    "purpose": self._usage_purpose,
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                }
            )
        return LLMResponse(message, usage)
