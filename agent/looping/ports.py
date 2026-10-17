from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.bus.event_bus import EventBus
    from agent.bus.queue import MessageBus
    from agent.memory.service2 import MemoryServiceV2
    from agent.provider import LLMProvider
    from agent.reliability import ReliabilityStore
    from agent.tools.base import ToolRegistry
    from core.memory.engine import MemoryEngine
    from proactive_v2.presence import PresenceStore
    from session.manager import SessionManager


# ── Config dataclasses（参数，不含服务对象）───────────────────────────────────


@dataclass
class LLMConfig:
    model: str = "deepseek-chat"
    light_model: str = ""
    max_iterations: int = 10
    max_tokens: int = 8192
    tool_timeout_seconds: float = 30.0
    tool_search_enabled: bool = False
    multimodal: bool = True
    vl_available: bool = False
    system_prompt: str = "You are a helpful assistant."


@dataclass
class MemoryConfig:
    window: int = 40

    @property
    def keep_count(self) -> int:
        """上下文携带条数，也是 consolidation 后 session 保留条数。"""
        aligned_window = max(4, ((max(1, self.window) + 3) // 4) * 4)
        return aligned_window // 2


# ── 服务对象分组（仅放对象，不放配置参数）──────────────────────────────────────


@dataclass
class LLMServices:
    """LLM provider services."""

    provider: LLMProvider
    light_provider: LLMProvider


@dataclass
class MemoryServices:
    engine: MemoryEngine | None = None


@dataclass
class SessionServices:
    session_manager: SessionManager
    presence: PresenceStore | None = None


@dataclass
class AgentLoopDeps:
    bus: "MessageBus"
    provider: "LLMProvider"
    tools: "ToolRegistry"
    session_manager: "SessionManager"
    workspace: Path
    event_bus: "EventBus | None" = None
    presence: "PresenceStore | None" = None
    light_provider: "LLMProvider | None" = None
    memory_service: "MemoryServiceV2 | None" = None
    reliability_store: "ReliabilityStore | None" = None
    llm_services: LLMServices | None = None
    memory_services: MemoryServices | None = None
    session_services: SessionServices | None = None

@dataclass
class AgentLoopConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
