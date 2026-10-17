from __future__ import annotations

import asyncio

import pytest

from agent.bus.event_bus import EventBus
from agent.bus.events import InboundMessage
from agent.bus.queue import MessageBus
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig, MemoryConfig
from agent.provider import LLMResponse, ToolCall
from agent.tools.base import BaseTool, ToolRegistry
from agent.tools.tool_search import ToolSearchTool
from session.manager import SessionManager


class _EchoTool(BaseTool):
    name = "echo"
    description = "echo text"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, text: str = "") -> str:
        return f"echo:{text}"


class _SlowTool(BaseTool):
    name = "slow"
    description = "slow test tool"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> str:
        await asyncio.sleep(0.05)
        return "finished"


class _ScriptedProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call-1", name="echo", arguments={"text": "hello"})
                ],
            )
        return LLMResponse(content="工具执行完成")


class _StreamingProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **kwargs) -> LLMResponse:
        self.calls += 1
        on_delta = kwargs["on_content_delta"]
        if self.calls == 1:
            await on_delta("准备调用工具")
            return LLMResponse(
                content="准备调用工具",
                tool_calls=[
                    ToolCall(id="stream-tool-1", name="echo", arguments={"text": "hi"})
                ],
            )
        await on_delta("最终")
        await on_delta("回答")
        return LLMResponse(content="最终回答")


class _MemoryAwareProvider:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def chat(self, **kwargs) -> LLMResponse:
        self.messages = kwargs["messages"]
        return LLMResponse(content="已参考相关记忆")


class _QueryMemoryService:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve_memory_context(self, query: str) -> str:
        self.queries.append(query)
        return "- [长期记忆] 用户喜欢跑步"

    async def extract_facts(self, user_message: str, assistant_message: str):
        return []


class _DeferredToolProvider:
    """依次搜索工具、调用刚加载的工具、输出最终答案。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        visible = {
            schema["function"]["name"] for schema in (kwargs.get("tools") or [])
        }
        if len(self.calls) == 1:
            assert visible == {"tool_search"}
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="search-1",
                        name="tool_search",
                        arguments={"query": "echo text"},
                    )
                ],
            )
        if len(self.calls) == 2:
            assert visible == {"tool_search", "echo"}
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="echo-1", name="echo", arguments={"text": "hello"})
                ],
            )
        return LLMResponse(content="动态工具执行完成")


class _DeleteConfirmationProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **kwargs) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="delete-1",
                        name="delete_file",
                        arguments={"path": "remove_me.txt"},
                    )
                ],
            )
        return LLMResponse(content="已收到删除结果")


@pytest.mark.asyncio
async def test_new_loop_runs_tools_and_persists_session(tmp_path) -> None:
    bus = MessageBus()
    events = EventBus()
    sessions = SessionManager(tmp_path)
    tools = ToolRegistry()
    tools.register(_EchoTool())
    provider = _ScriptedProvider()
    loop = AgentLoop(
        AgentLoopDeps(
            bus=bus,
            event_bus=events,
            provider=provider,  # type: ignore[arg-type]
            tools=tools,
            session_manager=sessions,
            workspace=tmp_path,
        ),
        AgentLoopConfig(
            llm=LLMConfig(
                model="test-model",
                max_iterations=3,
                max_tokens=256,
                system_prompt="test system",
            ),
            memory=MemoryConfig(window=8),
        ),
    )

    task = asyncio.create_task(loop.run())
    await bus.publish_inbound(
        InboundMessage(
            channel="api",
            sender_id="alice",
            chat_id="request-1",
            content="say hello",
            session_key_override="api:alice",
        )
    )
    outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=2)

    assert outbound.content == "工具执行完成"
    assert outbound.metadata["agent_loop"] == "new"
    assert outbound.metadata["tool_chain"][0]["calls"][0]["name"] == "echo"
    assert outbound.metadata["tool_chain"][0]["calls"][0]["status"] == "succeeded"
    assert outbound.metadata["tool_chain"][0]["calls"][0]["duration_ms"] >= 0
    assert outbound.metadata["tool_chain"][0]["calls"][0]["name"] == "echo"
    assert outbound.metadata["tool_chain"][0]["calls"][0]["status"] == "succeeded"
    assert outbound.metadata["tool_chain"][0]["calls"][0]["duration_ms"] >= 0
    assert len(provider.calls) == 2
    assert provider.calls[0]["max_tokens"] == 256
    assert provider.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "echo:hello",
    }

    loop.stop()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await events.close()
    sessions.close()

    reopened = SessionManager(tmp_path)
    history = reopened.get_or_create("api:alice").get_history(max_messages=10)
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "say hello"
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "工具执行完成"
    reopened.close()


@pytest.mark.asyncio
async def test_new_loop_emits_stream_deltas_and_tool_progress(tmp_path) -> None:
    bus = MessageBus()
    events = EventBus()
    sessions = SessionManager(tmp_path)
    tools = ToolRegistry()
    tools.register(_EchoTool())
    provider = _StreamingProvider()
    loop = AgentLoop(
        AgentLoopDeps(
            bus=bus,
            event_bus=events,
            provider=provider,  # type: ignore[arg-type]
            tools=tools,
            session_manager=sessions,
            workspace=tmp_path,
        ),
        AgentLoopConfig(
            llm=LLMConfig(model="test-model", max_iterations=3),
            memory=MemoryConfig(window=8),
        ),
    )
    stream_queue: asyncio.Queue[dict] = asyncio.Queue()
    message = InboundMessage(
        channel="api",
        sender_id="alice",
        chat_id="stream-request",
        content="stream this",
        metadata={"_stream_event_queue": stream_queue},
        session_key_override="api:alice",
    )

    outbound = await loop._process(message, dispatch_outbound=False)
    streamed = []
    while not stream_queue.empty():
        streamed.append(stream_queue.get_nowait())

    event_types = [event["type"] for event in streamed]
    assert outbound.content == "最终回答"
    assert event_types.count("answer_delta") == 3
    assert "answer_reset" in event_types
    assert "tool_started" in event_types
    assert "tool_finished" in event_types
    assert streamed[-1] == {
        "type": "answer_delta",
        "delta": "回答",
        "iteration": 2,
    }

    await events.close()
    sessions.close()
    loop.close()


@pytest.mark.asyncio
async def test_new_loop_retrieves_memory_with_current_user_query(tmp_path) -> None:
    bus = MessageBus()
    events = EventBus()
    sessions = SessionManager(tmp_path)
    provider = _MemoryAwareProvider()
    memory = _QueryMemoryService()
    loop = AgentLoop(
        AgentLoopDeps(
            bus=bus,
            event_bus=events,
            provider=provider,  # type: ignore[arg-type]
            tools=ToolRegistry(),
            session_manager=sessions,
            workspace=tmp_path,
            memory_service=memory,  # type: ignore[arg-type]
        ),
        AgentLoopConfig(
            llm=LLMConfig(model="test", system_prompt="system"),
            memory=MemoryConfig(window=8),
        ),
    )

    reply = await loop.process_direct("给我推荐跑步路线")

    assert reply == "已参考相关记忆"
    assert memory.queries == ["给我推荐跑步路线"]
    assert "用户喜欢跑步" in provider.messages[0]["content"]
    sessions.close()
    await events.close()


@pytest.mark.asyncio
async def test_tool_timeout_is_converted_to_recoverable_result(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tools = ToolRegistry()
    tools.register(_SlowTool())
    loop = AgentLoop(
        AgentLoopDeps(
            bus=MessageBus(),
            provider=_MemoryAwareProvider(),  # type: ignore[arg-type]
            tools=tools,
            session_manager=sessions,
            workspace=tmp_path,
        ),
        AgentLoopConfig(
            llm=LLMConfig(model="test", tool_timeout_seconds=0.01),
        ),
    )

    result = await loop._execute_tool("slow", {})

    assert "执行超时" in result
    sessions.close()


@pytest.mark.asyncio
async def test_tool_search_dynamically_loads_matched_schema(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    tools = ToolRegistry()
    tools.register(_EchoTool())
    tools.register(ToolSearchTool(tools), always_on=True)
    provider = _DeferredToolProvider()
    loop = AgentLoop(
        AgentLoopDeps(
            bus=MessageBus(),
            provider=provider,  # type: ignore[arg-type]
            tools=tools,
            session_manager=sessions,
            workspace=tmp_path,
        ),
        AgentLoopConfig(
            llm=LLMConfig(
                model="test",
                max_iterations=4,
                tool_search_enabled=True,
            )
        ),
    )

    reply = await loop.process_direct("echo hello")

    assert reply == "动态工具执行完成"
    assert len(provider.calls) == 3
    sessions.close()


@pytest.mark.asyncio
async def test_delete_file_waits_for_explicit_user_confirmation(tmp_path) -> None:
    from agent.tools.filesystem import register_filesystem_tools

    (tmp_path / "remove_me.txt").write_text("temporary", encoding="utf-8")
    sessions = SessionManager(tmp_path)
    tools = ToolRegistry()
    register_filesystem_tools(tools, tmp_path)
    provider = _DeleteConfirmationProvider()
    loop = AgentLoop(
        AgentLoopDeps(
            bus=MessageBus(),
            provider=provider,  # type: ignore[arg-type]
            tools=tools,
            session_manager=sessions,
            workspace=tmp_path,
        ),
        AgentLoopConfig(llm=LLMConfig(model="test", max_iterations=3)),
    )

    first = await loop.process_direct(
        "删除 remove_me.txt",
        session_key="cli:delete-test",
    )
    assert "确认删除" in first
    assert (tmp_path / "remove_me.txt").exists()

    second = await loop.process_direct(
        "确认删除",
        session_key="cli:delete-test",
    )
    assert "已删除文件" in second
    assert not (tmp_path / "remove_me.txt").exists()
    assert list((tmp_path / ".trash").rglob("remove_me.txt"))
    assert provider.calls == 1
    sessions.close()

