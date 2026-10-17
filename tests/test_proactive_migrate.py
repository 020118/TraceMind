"""proactive_v2 迁移验证测试。

覆盖：配置加载、装配层、记忆适配器、SessionManager/PresenceStore 集成、
以及一次真实 ProactiveLoop._tick() 的干净退出。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from bus.event_bus import EventBus  # noqa: F401
from core.memory.engine import MemoryQuery
from proactive_bootstrap import (
    MemoryRuntimeAdapter,
    SharedToolsAdapter,
    build_proactive_loop,
)
from proactive_v2.config import ProactiveConfig
from proactive_v2.config_loader import ProactiveConfigError, load_proactive_config


VALID_RAW = {
    "enabled": True,
    "profile": "daily",
    "target": {"channel": "console", "chat_id": "local"},
}


class _FakeLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, tools=None, tool_choice="auto"):
        self.calls += 1
        raise AssertionError("gate 应当在调用 LLM 之前拦截")


def _console_send(msg) -> None:
    raise AssertionError("gate 拦截后不应发送消息")


async def test_load_proactive_config_valid():
    cfg = load_proactive_config(VALID_RAW)
    assert isinstance(cfg, ProactiveConfig)
    assert cfg.enabled is True
    assert cfg.default_channel == "console"
    assert cfg.default_chat_id == "local"


async def test_load_proactive_config_rejects_unknown_root_key():
    bad = dict(VALID_RAW)
    bad["unknown_key"] = 1
    with pytest.raises(ProactiveConfigError):
        load_proactive_config(bad)


async def test_build_proactive_loop_disabled(tmp_path: Path):
    loop = build_proactive_loop(
        config_raw={"enabled": False},
        workspace=tmp_path,
        llm_client=_FakeLLMClient(),
        shared_registry=_make_registry(),
        event_bus=None,
        console_send=_console_send,
        model="test-model",
    )
    assert loop is None


def _make_registry():
    from agent.tools.base import BaseTool, ToolRegistry

    class DummyTool(BaseTool):
        name = "dummy"
        description = "dummy"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kwargs):
            return "ok"

    registry = ToolRegistry()
    registry.register(DummyTool())
    return registry


async def test_full_loop_tick_gate_blocks_without_chat_id(tmp_path: Path):
    raw = {
        "enabled": True,
        "profile": "daily",
        "target": {"channel": "console", "chat_id": ""},
    }
    loop = build_proactive_loop(
        config_raw=raw,
        workspace=tmp_path,
        llm_client=_FakeLLMClient(),
        shared_registry=_make_registry(),
        event_bus=None,
        console_send=_console_send,
        memory_service=None,
        model="test-model",
    )
    assert loop is not None
    assert (tmp_path / "PROACTIVE_CONTEXT.md").exists()
    score = await loop._tick()
    assert score is None or isinstance(score, float)


async def test_shared_tools_adapter(tmp_path: Path):
    adapter = SharedToolsAdapter(_make_registry())
    tool = adapter.get_tool("dummy")
    assert tool is not None
    assert await tool.execute() == "ok"
    assert adapter.get_tool("missing") is None
    assert adapter.get_mcp_server_names() == set()


class _FakeStore:
    def read_self(self) -> str:
        return "# SELF\n- 用户喜欢深夜工作"

    def read_long_term(self) -> str:
        return "# MEMORY\n- 用户正在开发 agent 项目\n- 用户偏好中文交流"

    def read_recent_context(self) -> str:
        return "- 今天讨论了 proactive 迁移"

    def get_memory_context(self) -> str:
        return "ctx"


class _FakeService:
    def __init__(self) -> None:
        self.store = _FakeStore()


async def test_memory_runtime_adapter_query_scoring():
    adapter = MemoryRuntimeAdapter(_FakeService())
    assert adapter.read_self().startswith("# SELF")

    result = await adapter.query(
        MemoryQuery(text="用户 agent 项目", limit=3)
    )
    assert len(result.records) >= 1
    top = result.records[0]
    assert "agent" in top.summary
    assert top.score > 0


async def test_session_manager_and_presence(tmp_path: Path):
    from proactive_v2.presence import PresenceStore
    from session.manager import SessionManager

    manager = SessionManager(tmp_path)
    presence = PresenceStore(manager._store)

    session = manager.get_or_create("console:local")
    session.add_message("user", "你好")
    await manager.append_messages(session, session.messages[-1:])

    presence.record_user_message("console:local")
    last_at = presence.get_last_user_at("console:local")
    assert isinstance(last_at, datetime)
    assert last_at.tzinfo is timezone.utc or last_at.utcoffset() is not None
