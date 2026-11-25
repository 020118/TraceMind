from __future__ import annotations

from pathlib import Path

import pytest

from agent.bus.events import InboundMessage
from agent.bus.queue import MessageBus
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, ContextConfig, LLMConfig
from agent.provider import LLMResponse
from agent.reliability import ReliabilityStore
from agent.reliability.journal import TASK_RUNNING, TASK_SUCCEEDED
from agent.tools.base import ToolRegistry
from session.manager import SessionManager


class _ReplyProvider:
    async def chat(self, **_kwargs) -> LLMResponse:
        return LLMResponse(
            content="traceable reply",
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
        )


def test_task_journal_persists_checkpoint_and_status(tmp_path: Path) -> None:
    store = ReliabilityStore(tmp_path / "reliability.db")
    task = store.create_task_run(
        session_key="api:alice",
        channel="api",
        chat_id="request-1",
        user_content="hello",
    )
    store.update_task_run(
        task.trace_id,
        phase="awaiting_model",
        iteration=1,
        checkpoint={"messages": [{"role": "user", "content": "hello"}]},
    )
    checkpointed = store.get_task_run(task.trace_id)
    assert checkpointed is not None
    assert checkpointed.status == TASK_RUNNING
    assert checkpointed.phase == "awaiting_model"
    assert checkpointed.checkpoint["messages"][0]["content"] == "hello"

    store.update_task_run(
        task.trace_id,
        status=TASK_SUCCEEDED,
        phase="completed",
        checkpoint={},
    )
    listed = store.list_task_runs(session_key="api:alice")
    assert listed[0].trace_id == task.trace_id
    assert listed[0].completed_at is not None
    assert store.stats()["tasks_succeeded"] == 1
    store.close()


@pytest.mark.asyncio
async def test_agent_turn_returns_trace_id_and_marks_task_succeeded(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)
    store = ReliabilityStore(tmp_path / "reliability.db")
    loop = AgentLoop(
        AgentLoopDeps(
            bus=MessageBus(),
            provider=_ReplyProvider(),
            tools=ToolRegistry(),
            session_manager=sessions,
            workspace=tmp_path,
            reliability_store=store,
        ),
        AgentLoopConfig(
            llm=LLMConfig(model="test", max_iterations=2),
            context=ContextConfig(auto_summary=False),
        ),
    )

    outbound = await loop._process(
        InboundMessage(
            channel="api",
            sender_id="alice",
            chat_id="alice",
            content="hello",
            session_key_override="api:alice",
        ),
        dispatch_outbound=False,
    )

    trace_id = str(outbound.metadata.get("trace_id", ""))
    assert trace_id.startswith("trace_")
    record = store.get_task_run(trace_id)
    assert record is not None
    assert record.status == TASK_SUCCEEDED
    assert record.phase == "completed"
    usage = store.model_usage_summary()
    assert usage["calls"] == 1
    assert usage["total_tokens"] == 14
    sessions.close()
    store.close()
