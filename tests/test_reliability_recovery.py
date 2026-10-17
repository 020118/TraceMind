from __future__ import annotations

import pytest

from agent.bus.queue import MessageBus
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig
from agent.provider import LLMResponse, ToolCall
from agent.reliability import ReliabilityStore
from agent.reliability.journal import CONFIRMED, RUNNING, SUCCEEDED
from agent.tools.base import ToolRegistry
from agent.tools.filesystem import register_filesystem_tools
from session.manager import SessionManager


class _DeleteProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **_kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="delete-call",
                    name="delete_file",
                    arguments={"path": "remove_me.txt"},
                )
            ],
        )


class _ReplyProvider:
    async def chat(self, **_kwargs) -> LLMResponse:
        return LLMResponse(content="normal reply")


def _build_loop(tmp_path, sessions, store, provider) -> AgentLoop:
    tools = ToolRegistry()
    register_filesystem_tools(tools, tmp_path)
    return AgentLoop(
        AgentLoopDeps(
            bus=MessageBus(),
            provider=provider,
            tools=tools,
            session_manager=sessions,
            workspace=tmp_path,
            reliability_store=store,
        ),
        AgentLoopConfig(llm=LLMConfig(model="test", max_iterations=2)),
    )


@pytest.mark.asyncio
async def test_pending_confirmation_survives_process_restart(tmp_path) -> None:
    (tmp_path / "remove_me.txt").write_text("temporary", encoding="utf-8")
    sessions1 = SessionManager(tmp_path)
    store1 = ReliabilityStore(tmp_path / "reliability.db")
    provider = _DeleteProvider()
    loop1 = _build_loop(tmp_path, sessions1, store1, provider)

    first = await loop1.process_direct("删除 remove_me.txt", session_key="cli:alice")
    pending = store1.get_pending_operation("cli:alice")
    assert "确认删除" in first
    assert pending is not None
    sessions1.close()
    store1.close()

    sessions2 = SessionManager(tmp_path)
    store2 = ReliabilityStore(tmp_path / "reliability.db")
    loop2 = _build_loop(tmp_path, sessions2, store2, _ReplyProvider())
    second = await loop2.process_direct("确认删除", session_key="cli:alice")

    assert "已删除文件" in second
    assert not (tmp_path / "remove_me.txt").exists()
    completed = store2.get_operation(pending.operation_id)
    assert completed is not None
    assert completed.status == SUCCEEDED
    sessions2.close()
    store2.close()


@pytest.mark.asyncio
async def test_running_delete_is_reconciled_after_restart(tmp_path) -> None:
    (tmp_path / "remove_me.txt").write_text("temporary", encoding="utf-8")
    store = ReliabilityStore(tmp_path / "reliability.db")
    pending = store.create_pending_operation(
        session_key="cli:alice",
        tool_name="delete_file",
        arguments={"path": "remove_me.txt"},
    )
    store.transition_operation(
        pending.operation_id,
        to_status=CONFIRMED,
        allowed_from={pending.status},
    )
    store.transition_operation(
        pending.operation_id,
        to_status=RUNNING,
        allowed_from={CONFIRMED},
    )

    sessions = SessionManager(tmp_path)
    loop = _build_loop(tmp_path, sessions, store, _ReplyProvider())
    reply = await loop.process_direct("继续", session_key="cli:bob")

    assert reply == "normal reply"
    assert not (tmp_path / "remove_me.txt").exists()
    recovered = store.get_operation(pending.operation_id)
    assert recovered is not None
    assert recovered.status == SUCCEEDED
    assert list((tmp_path / ".trash" / pending.operation_id).rglob("remove_me.txt"))
    sessions.close()
    store.close()


@pytest.mark.asyncio
async def test_telegram_outbox_survives_bus_restart(tmp_path) -> None:
    from agent.bus.events import OutboundMessage

    store1 = ReliabilityStore(tmp_path / "reliability.db")
    bus1 = MessageBus()
    bus1.configure_outbox(store1, durable_channels={"telegram"})
    await bus1.publish_outbound(
        OutboundMessage(channel="telegram", chat_id="chat-1", content="important")
    )
    store1.close()

    store2 = ReliabilityStore(tmp_path / "reliability.db")
    bus2 = MessageBus()
    bus2.configure_outbox(store2, durable_channels={"telegram"})
    restored = await bus2.restore_pending_outbound()
    message = await bus2.consume_outbound()

    assert restored == 1
    assert message.content == "important"
    outbox_id = message.metadata["_outbox_id"]
    assert outbox_id
    await bus2.mark_outbound_sent(message)
    assert store2.list_pending_outbound() == []
    assert store2.get_outbound_status(outbox_id) == "SENT"
    store2.close()


@pytest.mark.asyncio
async def test_delete_result_and_telegram_outbox_are_committed_together(
    tmp_path,
) -> None:
    (tmp_path / "remove_me.txt").write_text("temporary", encoding="utf-8")
    sessions = SessionManager(tmp_path)
    store = ReliabilityStore(tmp_path / "reliability.db")
    loop = _build_loop(tmp_path, sessions, store, _DeleteProvider())

    await loop.process_direct(
        "删除 remove_me.txt",
        session_key="telegram:chat-1",
        channel="telegram",
        chat_id="chat-1",
        dispatch_outbound=False,
    )
    result = await loop.process_direct(
        "确认删除",
        session_key="telegram:chat-1",
        channel="telegram",
        chat_id="chat-1",
        dispatch_outbound=False,
    )

    assert "已删除文件" in result
    pending_messages = store.list_pending_outbound()
    assert len(pending_messages) == 1
    assert pending_messages[0]["channel"] == "telegram"
    assert "已删除文件" in pending_messages[0]["content"]
    sessions.close()
    store.close()
