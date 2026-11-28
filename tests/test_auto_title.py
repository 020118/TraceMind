from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agent.bus.queue import MessageBus
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig
from agent.provider import LLMResponse
from agent.reliability import ReliabilityStore
from agent.tools.base import ToolRegistry
from session.manager import SessionManager


class _TitleProvider:
    def __init__(self, reply: str = "Python 异步爬虫优化", *, fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls = 0

    async def chat(self, **_kwargs: Any) -> LLMResponse:
        self.calls += 1
        if self.fail:
            raise RuntimeError("network error")
        return LLMResponse(content=self.reply)


def _build_loop(tmp_path: Path, light_provider: Any) -> tuple[AgentLoop, SessionManager, ReliabilityStore]:
    sessions = SessionManager(tmp_path)
    store = ReliabilityStore(tmp_path / "reliability.db")
    loop = AgentLoop(
        AgentLoopDeps(
            bus=MessageBus(),
            provider=_TitleProvider("主回答内容"),
            tools=ToolRegistry(),
            session_manager=sessions,
            workspace=tmp_path,
            reliability_store=store,
            light_provider=light_provider,
        ),
        AgentLoopConfig(llm=LLMConfig(model="main-model", max_iterations=2)),
    )
    return loop, sessions, store


def test_auto_title_cleans_punctuation_and_persists(tmp_path: Path):
    light = _TitleProvider("《Docker 部署排障指南》。")
    loop, sessions, store = _build_loop(tmp_path, light)

    async def _run():
        sessions.get_or_create("api:s1")
        sessions.save(sessions.get_or_create("api:s1"))
        title = await loop._generate_session_title(
            "api:s1",
            "帮我排查 Docker 启动失败",
            "请检查 qdrant 健康检查状态。",
        )
        assert title == "Docker 部署排障指南"
        reloaded = sessions.get_or_create("api:s1")
        assert reloaded.metadata.get("title") == "Docker 部署排障指南"

    try:
        asyncio.run(_run())
    finally:
        sessions.close()
        store.close()


def test_auto_title_does_not_overwrite_manual_title(tmp_path: Path):
    light = _TitleProvider("自动生成的新标题")
    loop, sessions, store = _build_loop(tmp_path, light)

    async def _run():
        s = sessions.get_or_create("api:s2")
        sessions.save(s)
        sessions.update_session_metadata("api:s2", title="用户手动标题")
        title = await loop._generate_session_title("api:s2", "你好", "你好！")
        assert title is None
        assert sessions.get_or_create("api:s2").metadata.get("title") == "用户手动标题"

    try:
        asyncio.run(_run())
    finally:
        sessions.close()
        store.close()


def test_auto_title_degrades_silently_on_provider_error(tmp_path: Path):
    light = _TitleProvider(fail=True)
    loop, sessions, store = _build_loop(tmp_path, light)

    async def _run():
        s = sessions.get_or_create("api:s3")
        sessions.save(s)
        title = await loop._generate_session_title("api:s3", "测试", "回复")
        assert title is None
        assert not sessions.get_or_create("api:s3").metadata.get("title")

    try:
        asyncio.run(_run())
    finally:
        sessions.close()
        store.close()
