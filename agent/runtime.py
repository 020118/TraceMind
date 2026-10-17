"""Reusable application runtime shared by CLI, Telegram and FastAPI."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from agent.bus.event_bus import EventBus
from agent.bus.events import LifecycleEvent, LifecyclePhase
from agent.bus.queue import MessageBus
from agent.channels.api import ApiChannel
from agent.channels.console import console_send, run_console_channel
from agent.channels.dispatcher import SendFn, outbound_dispatcher
from agent.config_models import AgentConfig
from agent.core.events import AfterSendEvent
from agent.lifecycle.manager import LifecycleManager
from agent.lifecycle.types import AfterTurnCtx, BeforeTurnCtx
from agent.llm.client import LLMClient
from agent.looping.core import AgentLoop
from agent.looping.ports import (
    AgentLoopConfig,
    AgentLoopDeps,
    LLMConfig as LoopLLMConfig,
    MemoryConfig as LoopMemoryConfig,
)
from agent.memory.service2 import MemoryServiceV2, create_memory_service_v2
from agent.provider import LLMProvider
from agent.rag.service import RAGService, create_rag_service
from agent.reliability import ReliabilityStore
from agent.tools.amap import register_all_amap_tools as register_amap_direct
from agent.tools.base import BaseTool, ToolRegistry
from agent.tools.code_execution import register_code_execution_tools
from agent.tools.filesystem import register_filesystem_tools
from agent.tools.mcp_client import register_mcp_server
from agent.tools.mcp_stdio_client import register_mcp_stdio
from agent.tools.knowledge import register_knowledge_tools
from agent.tools.tool_search import ToolSearchTool
from agent.tools.weather import GetWeather
from agent.tools.web_search import WebFetch, WebSearch
from proactive_bootstrap import build_proactive_loop, register_telegram_push
from session.manager import SessionManager

logger = logging.getLogger(__name__)


class ExampleEcho(BaseTool):
    name = "echo"
    description = "回声工具：返回你传给它的内容"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要回声的文本"},
        },
        "required": ["text"],
    }

    async def execute(self, text: str = "") -> str:
        return f"回声: {text}"


def setup_tools(
    registry: ToolRegistry,
    workspace: Path | None = None,
    *,
    tool_timeout_seconds: float = 30.0,
) -> None:
    registry.register(ExampleEcho())
    registry.register(GetWeather(), aliases=("天气查询", "天气预报"))
    registry.register(WebSearch(), aliases=("网络搜索", "网页搜索", "查资料"))
    registry.register(WebFetch(), aliases=("读取网页", "抓取网页", "打开网址"))
    if workspace is not None:
        register_filesystem_tools(registry, workspace)
        register_code_execution_tools(
            registry,
            workspace,
            tool_timeout_seconds=tool_timeout_seconds,
        )
    registry.register(
        ToolSearchTool(registry),
        always_on=True,
        aliases=("工具搜索", "能力发现", "工具目录", "tool discovery"),
    )
    logger.info("工具注册完成，共 %d 个工具", registry.count)


def setup_event_handlers(events: EventBus) -> None:
    async def sensitive_filter(ctx: BeforeTurnCtx) -> BeforeTurnCtx | None:
        blocked_words = ("敏感词1", "敏感词2")
        if any(word in ctx.content for word in blocked_words):
            logger.warning("拦截含敏感词的消息: %s", ctx.content[:30])
            return None
        return ctx

    async def log_llm_response(ctx: AfterTurnCtx) -> None:
        logger.info(
            "[OBSERVE] channel=%s, query_len=%d, resp_len=%d",
            ctx.channel,
            len(ctx.content),
            len(ctx.reply),
        )

    async def log_fanout(event: AfterSendEvent) -> None:
        logger.info("[FANOUT] channel=%s content=%s...", event.channel, event.content[:20])

    async def log_lifecycle(event: LifecycleEvent) -> None:
        logger.info("[LIFECYCLE] %s: %s", event.phase.value, event.message)

    events.on(BeforeTurnCtx, sensitive_filter)
    events.on(AfterTurnCtx, log_llm_response)
    events.on(AfterSendEvent, log_fanout)
    events.on(LifecycleEvent, log_lifecycle)


class AgentRuntime:
    """Owns application resources and background tasks.

    `start()` and `stop()` are intentionally explicit so FastAPI's lifespan,
    the CLI and tests can all manage the same runtime safely.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        enable_console: bool = False,
        enable_telegram: bool = True,
        enable_proactive: bool = True,
    ) -> None:
        self.config = config
        self.enable_console = enable_console
        self.enable_telegram = enable_telegram
        self.enable_proactive = enable_proactive
        self.workspace = Path(config.runtime.workspace).expanduser().resolve()

        self.bus = MessageBus()
        self.events = EventBus()
        self.lifecycle = LifecycleManager(self.events)
        self.tools = ToolRegistry()
        self.api_channel = ApiChannel(self.bus)
        self.reliability_store: ReliabilityStore | None = None

        self.llm: LLMClient | None = None
        self.agent_loop: AgentLoop | None = None
        self.session_manager: SessionManager | None = None
        self.memory_service: MemoryServiceV2 | None = None
        self.rag_service: RAGService | None = None
        self.proactive_loop: Any = None
        self.telegram_channel: Any = None
        self.mcp_amap_server: Any = None
        self._tasks: list[asyncio.Task[Any]] = []
        self.ready = False
        self.started = False

    @classmethod
    def from_config_path(
        cls,
        config_path: str = "config.toml",
        *,
        workspace: str | None = None,
        enable_console: bool = False,
        enable_telegram: bool = True,
        enable_proactive: bool = True,
    ) -> "AgentRuntime":
        config = AgentConfig.load(config_path)
        workspace_override = workspace or os.environ.get("AGENT_WORKSPACE")
        if workspace_override:
            config.runtime.workspace = workspace_override
        return cls(
            config,
            enable_console=enable_console,
            enable_telegram=enable_telegram,
            enable_proactive=enable_proactive,
        )

    async def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.workspace.mkdir(parents=True, exist_ok=True)

        try:
            self.reliability_store = ReliabilityStore(
                self.workspace / ".agent_state" / "reliability.db"
            )
            self.bus.configure_outbox(
                self.reliability_store,
                durable_channels={"telegram"},
            )
            setup_tools(
                self.tools,
                self.workspace,
                tool_timeout_seconds=self.config.tool_timeout_seconds,
            )
            await self._initialize_memory()
            await self._initialize_rag()
            await self._register_external_tools()
            setup_event_handlers(self.events)
            await self.lifecycle.start()

            main_cfg = self.config.llm.main
            if main_cfg is None or not main_cfg.api_key:
                raise ValueError(
                    "缺少 LLM API Key，请在 config.toml 的 [llm.main] 中设置 api_key"
                )
            self.llm = LLMClient(
                api_key=main_cfg.api_key,
                base_url=main_cfg.base_url,
                model=main_cfg.model,
            )
            self.session_manager = SessionManager(self.workspace)
            loop_provider = LLMProvider(self.llm)
            light_provider = self._build_light_provider(loop_provider)
            self.agent_loop = AgentLoop(
                AgentLoopDeps(
                    bus=self.bus,
                    event_bus=self.events,
                    provider=loop_provider,
                    light_provider=light_provider,
                    tools=self.tools,
                    session_manager=self.session_manager,
                    workspace=self.workspace,
                    memory_service=self.memory_service,
                    reliability_store=self.reliability_store,
                ),
                AgentLoopConfig(
                    llm=LoopLLMConfig(
                        model=main_cfg.model,
                        light_model=(
                            self.config.llm.fast.model
                            if self.config.llm.fast is not None
                            else main_cfg.model
                        ),
                        max_iterations=self.config.max_iterations,
                        max_tokens=self.config.max_tokens,
                        tool_timeout_seconds=self.config.tool_timeout_seconds,
                        tool_search_enabled=True,
                        system_prompt=self.config.system_prompt,
                    ),
                    memory=LoopMemoryConfig(window=self.config.memory_window),
                ),
            )

            if self.enable_proactive:
                self.proactive_loop = build_proactive_loop(
                    config_raw=self.config.proactive,
                    workspace=self.workspace,
                    llm_client=self.llm,
                    shared_registry=self.tools,
                    event_bus=self.events,
                    console_send=console_send,
                    memory_service=self.memory_service,
                    model=main_cfg.model,
                    max_tokens=self.config.max_tokens,
                )

            handlers: dict[str, SendFn] = {"api": self.api_channel.send}
            if self.enable_console:
                handlers["console"] = console_send

            await self._initialize_telegram(handlers)

            if "telegram" in handlers:
                restored = await self.bus.restore_pending_outbound()
                if restored:
                    logger.info("恢复待发送出站消息 count=%d", restored)

            self._spawn(
                self.agent_loop.run(),
                "agent-loop",
            )
            self._spawn(
                outbound_dispatcher(
                    self.bus,
                    handlers,
                    dead_letter_path=(
                        self.workspace / "dead_letter" / "outbound.jsonl"
                    ),
                ),
                "outbound-dispatcher",
            )

            if self.enable_console:
                self._spawn(run_console_channel(self.bus), "console-channel")
            if self.telegram_channel is not None:
                self._spawn(self.telegram_channel.start(), "telegram-channel")
            if self.proactive_loop is not None:
                self._spawn(self.proactive_loop.run(), "proactive-loop")

            self.ready = True
            logger.info("AgentRuntime 已就绪 workspace=%s", self.workspace)
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        if not self.started and not self._tasks:
            return
        self.ready = False

        if self.agent_loop is not None:
            self.agent_loop.stop()
            drained = await self.agent_loop.drain(timeout=5.0)
            if not drained:
                logger.warning("等待执行中回合结束超时，将取消剩余任务")
        if self.proactive_loop is not None:
            self.proactive_loop.stop()
        if self.telegram_channel is not None:
            try:
                await self.telegram_channel.stop()
            except Exception:
                logger.exception("Telegram 关闭失败")

        await self.api_channel.close()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self.mcp_amap_server is not None:
            await self.mcp_amap_server.close()
            self.mcp_amap_server = None
        if self.memory_service is not None:
            await self.memory_service.close()
            self.memory_service = None
        if self.rag_service is not None:
            await self.rag_service.close()
            self.rag_service = None
        if self.session_manager is not None:
            self.session_manager.close()
            self.session_manager = None
        self.agent_loop = None

        if self.reliability_store is not None:
            self.reliability_store.close()
            self.reliability_store = None

        try:
            await self.lifecycle.stop()
        finally:
            await self.events.close()
        self.started = False
        logger.info("AgentRuntime 已安全关闭")

    def status(self) -> dict[str, Any]:
        snapshot = {
            "ready": self.ready,
            "llm": self.llm is not None,
            "memory": self.memory_service is not None,
            "rag": self.rag_service is not None,
            "proactive": self.proactive_loop is not None,
            "telegram": self.telegram_channel is not None,
            "inbound_queue": self.bus.inbound_size,
            "outbound_queue": self.bus.outbound_size,
            "pending_http_requests": self.api_channel.pending_count,
            "agent_loop": "new" if self.agent_loop is not None else "stopped",
        }
        if self.reliability_store is not None:
            snapshot["reliability"] = self.reliability_store.stats()
        if self.rag_service is not None:
            snapshot["rag_vector_store"] = {
                "name": (
                    self.rag_service.vector_store.name
                    if self.rag_service.vector_store is not None
                    else "sqlite_exact"
                ),
                "available": bool(
                    self.rag_service.vector_store is not None
                    and self.rag_service.vector_store.available
                ),
            }
        return snapshot

    def _spawn(self, coroutine: Any, name: str) -> None:
        task = asyncio.create_task(coroutine, name=name)
        task.add_done_callback(self._log_background_failure)
        self._tasks.append(task)

    @staticmethod
    def _log_background_failure(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.error("后台任务异常退出 task=%s", task.get_name(), exc_info=exc)

    async def _register_external_tools(self) -> None:
        mcp_url = os.environ.get("MCP_SERVER_URL", "")
        mcp_token = os.environ.get("MCP_SERVER_TOKEN", "")
        if mcp_url:
            await register_mcp_server(self.tools, mcp_url, mcp_token)

        amap_api_key = os.environ.get("AMAP_API_KEY", "")
        if amap_api_key:
            register_amap_direct(self.tools, api_key=amap_api_key)

        mcp_amap_key = os.environ.get("MCP_AMAP_API_KEY", "")
        if mcp_amap_key and not amap_api_key:
            try:
                self.mcp_amap_server = await register_mcp_stdio(
                    self.tools,
                    command="uvx",
                    args=["amap-mcp-server"],
                    env={"AMAP_MAPS_API_KEY": mcp_amap_key},
                )
            except Exception as exc:
                logger.warning("高德 MCP stdio 连接失败: %s", exc)

    async def _initialize_memory(self) -> None:
        if not self.config.memory.enabled:
            return
        try:
            fast_cfg = self.config.llm.fast
            embedding_cfg = self.config.memory.embedding
            retrieval_cfg = self.config.memory.retrieval
            self.memory_service = create_memory_service_v2(
                workspace=self.workspace,
                fast_llm_config={
                    "api_key": fast_cfg.api_key,
                    "base_url": fast_cfg.base_url,
                    "model": fast_cfg.model,
                }
                if fast_cfg is not None and fast_cfg.api_key
                else None,
                embedding_config={
                    "api_key": embedding_cfg.api_key,
                    "base_url": embedding_cfg.base_url,
                    "model": embedding_cfg.model,
                },
                retrieval_config={
                    "top_k": retrieval_cfg.top_k,
                    "score_threshold": retrieval_cfg.score_threshold,
                    "semantic_weight": retrieval_cfg.semantic_weight,
                    "keyword_weight": retrieval_cfg.keyword_weight,
                },
            )
            await self.memory_service.store.initialize()
        except Exception as exc:
            logger.warning("记忆系统初始化失败（核心聊天仍可使用）: %s", exc)
            self.memory_service = None

    async def _initialize_rag(self) -> None:
        if not self.config.rag.enabled:
            return
        try:
            rag_cfg = self.config.rag
            embedding_cfg = rag_cfg.embedding
            self.rag_service = create_rag_service(
                workspace=self.workspace,
                embedding_config={
                    "api_key": embedding_cfg.api_key,
                    "base_url": embedding_cfg.base_url,
                    "model": embedding_cfg.model,
                },
                chunk_size=rag_cfg.chunk_size,
                chunk_overlap=rag_cfg.chunk_overlap,
                top_k=rag_cfg.top_k,
                rrf_k=rag_cfg.rrf_k,
                vector_store_config={
                    "provider": rag_cfg.vector_store.provider,
                    "url": rag_cfg.vector_store.url,
                    "api_key": rag_cfg.vector_store.api_key,
                    "collection": rag_cfg.vector_store.collection,
                    "timeout_seconds": rag_cfg.vector_store.timeout_seconds,
                    "read_from_qdrant": rag_cfg.vector_store.read_from_qdrant,
                    "shadow_evaluation": rag_cfg.vector_store.shadow_evaluation,
                    "sync_on_start": rag_cfg.vector_store.sync_on_start,
                },
            )
            await self.rag_service.initialize()
            register_knowledge_tools(self.tools, self.rag_service)
            logger.info(
                "RAG 知识库已就绪 path=%s semantic=%s",
                self.workspace / "knowledge" / "rag.db",
                self.rag_service.embedder is not None,
            )
        except Exception as exc:
            logger.warning("RAG 知识库初始化失败（核心聊天仍可使用）: %s", exc)
            if self.rag_service is not None:
                await self.rag_service.close()
            self.rag_service = None

    def _build_light_provider(self, fallback: LLMProvider) -> LLMProvider:
        fast_cfg = self.config.llm.fast
        if fast_cfg is None or not fast_cfg.api_key or not fast_cfg.model:
            return fallback
        return LLMProvider(
            LLMClient(
                api_key=fast_cfg.api_key,
                base_url=fast_cfg.base_url,
                model=fast_cfg.model,
            )
        )

    async def _initialize_telegram(self, handlers: dict[str, SendFn]) -> None:
        tg_cfg = self.config.channels.telegram
        if not self.enable_telegram or tg_cfg is None or not tg_cfg.token:
            return

        from agent.channels.telegram import TelegramChannel

        self.telegram_channel = TelegramChannel(tg_cfg.token, self.bus, tg_cfg.allow_from)
        handlers["telegram"] = self.telegram_channel.send
        if self.proactive_loop is not None:
            register_telegram_push(self.proactive_loop._push, self.telegram_channel.send)
