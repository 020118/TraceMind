"""proactive_v2 宿主装配层。

把 TraceMind 现有组件（LLMClient、ToolRegistry、MemoryServiceV2、console/telegram 通道）
适配成 proactive_v2.ProactiveLoop 需要的依赖，并完成整体组装。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.host_tools.base import Tool
from agent.host_tools.message_push import MessagePushTool
from agent.llm.client import LLMClient
from agent.memory.service2 import MemoryServiceV2
from agent.provider import LLMProvider
from agent.tools.base import BaseTool, ToolRegistry
from bus.event_bus import EventBus
from core.memory.engine import MemoryQuery, MemoryQueryResult, MemoryRecord
from proactive_v2.config import ProactiveConfig
from proactive_v2.config_loader import load_proactive_config
from proactive_v2.loop import ProactiveLoop
from proactive_v2.presence import PresenceStore
from session.manager import SessionManager

logger = logging.getLogger(__name__)


class HostToolAdapter(Tool):
    """把 TraceMind 的 BaseTool 包装成 host_tools 的 Tool 接口"""

    def __init__(self, inner: BaseTool) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def description(self) -> str:
        return self._inner.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._inner.parameters

    async def execute(self, **kwargs: Any) -> str:
        return await self._inner.execute(**kwargs)


class SharedToolsAdapter:
    """drift 工具注册表只消费 get_tool / get_mcp_server_names 两个方法"""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._cache: dict[str, HostToolAdapter | None] = {}

    def get_tool(self, name: str) -> HostToolAdapter | None:
        if name not in self._cache:
            inner = self._registry.get(name)
            self._cache[name] = HostToolAdapter(inner) if inner is not None else None
        return self._cache[name]

    def get_mcp_server_names(self) -> set[str]:
        return set()


class MemoryRuntimeAdapter:
    """实现 proactive_v2 需要的 MemoryRetrievalApi + MemoryProfileApi 双能力。

    检索基于六层 Markdown 记忆（MEMORY.md / RECENT_CONTEXT.md）做关键词重合度打分。
    """

    _TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")

    def __init__(self, memory_service: MemoryServiceV2 | None) -> None:
        self._service = memory_service

    @property
    def available(self) -> bool:
        return self._service is not None

    def _store(self) -> Any:
        return self._service.store if self._service is not None else None

    def read_self(self) -> str:
        store = self._store()
        return store.read_self() if store is not None else ""

    def read_long_term(self) -> str:
        store = self._store()
        return store.read_long_term() if store is not None else ""

    def read_recent_context(self) -> str:
        store = self._store()
        return store.read_recent_context() if store is not None else ""

    def get_memory_context(self) -> str:
        store = self._store()
        return store.get_memory_context() if store is not None else ""

    def has_long_term_memory(self) -> bool:
        return bool(self.read_long_term().strip())

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        return set(cls._TOKEN_RE.findall(text.lower()))

    @staticmethod
    def _iter_lines(text: str) -> list[str]:
        return [line.strip("-• \t") for line in text.splitlines() if len(line.strip("-• \t")) >= 8]

    async def query(self, request: MemoryQuery) -> MemoryQueryResult:
        tokens = self._tokenize(request.text or "")
        if not tokens or not self.available:
            return MemoryQueryResult()

        scored: list[tuple[float, int, str]] = []
        for source_kind, text in (
            ("long_term", self.read_long_term()),
            ("recent_context", self.read_recent_context()),
        ):
            for idx, line in enumerate(self._iter_lines(text)):
                line_tokens = self._tokenize(line)
                overlap = len(tokens & line_tokens)
                if overlap > 0:
                    score = overlap / max(len(tokens), len(line_tokens))
                    scored.append((score, idx, f"[{source_kind}] {line}"))

        scored.sort(key=lambda item: (-item[0], item[1]))
        limit = max(1, int(getattr(request, "limit", 5) or 5))
        records = [
            MemoryRecord(
                id=f"mem-{idx}",
                kind="fact",
                summary=line,
                score=round(score, 4),
                engine_kind="markdown_keyword",
            )
            for idx, (score, _pos, line) in enumerate(scored[:limit])
        ]
        text_block = "\n".join(f"- {r.summary}" for r in records)
        return MemoryQueryResult(text_block=text_block, records=records)


def build_push_tool(console_send: Any, workspace: Path) -> MessagePushTool:
    push_tool = MessagePushTool()

    async def _send_console(chat_id: str, message: str) -> None:
        from agent.bus.events import OutboundMessage

        await console_send(
            OutboundMessage(channel="console", chat_id=chat_id, content=message)
        )

    push_tool.register_channel("console", text=_send_console)
    logger.info("[proactive_bootstrap] console 推送通道已注册 (workspace=%s)", workspace)
    return push_tool


def register_telegram_push(push_tool: MessagePushTool, tg_send: Any) -> None:
    async def _send_telegram(chat_id: str, message: str) -> None:
        from agent.bus.events import OutboundMessage

        await tg_send(OutboundMessage(channel="telegram", chat_id=chat_id, content=message))

    push_tool.register_channel("telegram", text=_send_telegram)


def load_config_section(proactive_raw: dict[str, Any]) -> ProactiveConfig:
    return load_proactive_config(proactive_raw)


def _register_presence_recorder(event_bus: EventBus, presence: PresenceStore) -> None:
    from agent.lifecycle.types import BeforeTurnCtx

    async def _record(ctx: BeforeTurnCtx) -> BeforeTurnCtx:
        if ctx.session_key:
            presence.record_user_message(ctx.session_key)
        return ctx  # 必须返回 ctx：EventBus.emit 拦截链把返回 None 当作"拦截终止"，会吃掉所有用户消息

    event_bus.on(BeforeTurnCtx, _record)


def build_proactive_loop(
    *,
    config_raw: dict[str, Any],
    workspace: Path,
    llm_client: LLMClient,
    shared_registry: ToolRegistry,
    event_bus: EventBus | None,
    console_send: Any,
    memory_service: MemoryServiceV2 | None = None,
    model: str = "",
    max_tokens: int = 1024,
) -> ProactiveLoop | None:
    """按 config.toml 的 [proactive] 节装配 ProactiveLoop；未启用时返回 None。"""
    if not config_raw or not bool(config_raw.get("enabled", False)):
        logger.info("[proactive_bootstrap] proactive 未启用，跳过")
        return None

    cfg = load_proactive_config(config_raw)
    workspace.mkdir(parents=True, exist_ok=True)

    session_manager = SessionManager(workspace)
    presence = PresenceStore(session_manager._store)
    provider = LLMProvider(llm_client)
    push_tool = build_push_tool(console_send, workspace)
    memory_runtime = MemoryRuntimeAdapter(memory_service)

    if event_bus is not None:
        _register_presence_recorder(event_bus, presence)

    loop = ProactiveLoop(
        session_manager=session_manager,
        provider=provider,
        push_tool=push_tool,
        config=cfg,
        model=model,
        max_tokens=max_tokens,
        state_path=workspace / "proactive.db",
        memory_store=memory_runtime if memory_runtime.available else None,
        presence=presence,
        shared_tools=SharedToolsAdapter(shared_registry),
        event_bus=event_bus,
    )
    logger.info(
        "[proactive_bootstrap] ProactiveLoop 装配完成 target=%s:%s profile=%s",
        cfg.default_channel,
        cfg.default_chat_id,
        config_raw.get("profile", ""),
    )
    return loop


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
