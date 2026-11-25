from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent.bus.events import InboundMessage, OutboundMessage
from agent.bus.queue import MessageBus
from agent.channels.api import ApiChannel
from agent.llm.client import LLMClient
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig
from agent.provider import LLMResponse
from agent.tools.base import ToolRegistry
from api.app import create_app
from session.manager import SessionManager


class _FakeStreamChunk:
    def __init__(self, *, content: str | None = None, reasoning_content: str | None = None):
        self.choices = [
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=None,
                )
            )
        ]
        self.usage = None


class _AsyncChunkIter:
    def __init__(self, chunks):
        self._iter = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


@pytest.mark.asyncio
async def test_llm_client_streams_reasoning_and_content():
    chunks = [
        _FakeStreamChunk(reasoning_content="先分析"),
        _FakeStreamChunk(reasoning_content="问题"),
        _FakeStreamChunk(content="最终"),
        _FakeStreamChunk(content="答案"),
    ]

    async def fake_create(**kwargs):
        assert kwargs.get("stream") is True
        return _AsyncChunkIter(chunks)

    client = LLMClient(api_key="k", base_url="http://localhost", model="test-model")
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    reasoning_deltas: list[str] = []
    content_deltas: list[str] = []

    async def on_r(delta: str) -> None:
        reasoning_deltas.append(delta)

    async def on_c(delta: str) -> None:
        content_deltas.append(delta)

    resp = await client.chat(
        messages=[{"role": "user", "content": "hi"}],
        on_content_delta=on_c,
        on_reasoning_delta=on_r,
    )
    assert reasoning_deltas == ["先分析", "问题"]
    assert content_deltas == ["最终", "答案"]
    assert resp.reasoning_content == "先分析问题"
    assert resp.content == "最终答案"


@pytest.mark.asyncio
async def test_agent_loop_concurrent_across_sessions_and_fast_mode_routing(tmp_path: Path):
    bus = MessageBus()
    sm = SessionManager(tmp_path)
    tools = ToolRegistry()

    entered: list[str] = []
    gate = asyncio.Event()
    fast_called: list[str] = []

    class _SlowMainProvider:
        async def chat(self, **kwargs) -> LLMResponse:
            user_msg = kwargs["messages"][-1]["content"]
            entered.append(str(user_msg))
            if len(entered) >= 2:
                gate.set()
            await asyncio.wait_for(gate.wait(), timeout=2.0)
            on_r = kwargs.get("on_reasoning_delta")
            if on_r:
                await on_r("思考片段")
            return LLMResponse(content=f"main:{user_msg}", reasoning_content="思考片段")

    class _FastProvider:
        async def chat(self, **kwargs) -> LLMResponse:
            fast_called.append(str(kwargs.get("model")))
            assert kwargs.get("disable_thinking") is True
            return LLMResponse(content="fast-ok")

    loop = AgentLoop(
        AgentLoopDeps(
            bus=bus,
            provider=_SlowMainProvider(),
            light_provider=_FastProvider(),
            tools=tools,
            session_manager=sm,
            workspace=tmp_path,
        ),
        AgentLoopConfig(
            llm=LLMConfig(model="main-m", light_model="fast-m"),
            max_concurrent_turns=4,
        ),
    )

    runner = asyncio.create_task(loop.run())
    try:
        # 1. Cross-session concurrency: both sessions must enter provider.chat simultaneously.
        await bus.publish_inbound(
            InboundMessage(channel="api", sender_id="u1", chat_id="r1", content="A", session_key_override="api:s1")
        )
        await bus.publish_inbound(
            InboundMessage(channel="api", sender_id="u2", chat_id="r2", content="B", session_key_override="api:s2")
        )
        out1 = await asyncio.wait_for(bus.consume_outbound(), timeout=3.0)
        out2 = await asyncio.wait_for(bus.consume_outbound(), timeout=3.0)
        assert {out1.content, out2.content} == {"main:A", "main:B"}
        assert out1.metadata.get("reasoning_content") == "思考片段"

        # 2. Fast mode routes to light_provider + light_model + disable_thinking=True.
        await bus.publish_inbound(
            InboundMessage(
                channel="api",
                sender_id="u3",
                chat_id="r3",
                content="quick",
                metadata={"mode": "fast"},
                session_key_override="api:s3",
            )
        )
        out3 = await asyncio.wait_for(bus.consume_outbound(), timeout=3.0)
        assert out3.content == "fast-ok"
        assert fast_called == ["fast-m"]
    finally:
        loop.stop()
        await loop.drain(timeout=2.0)
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
        loop.close()
        sm.close()


def test_session_patch_truncate_and_chat_attachments_api(tmp_path: Path):
    bus = MessageBus()
    sm = SessionManager(tmp_path)
    api_channel = ApiChannel(bus)

    class _FakeRuntime:
        def __init__(self) -> None:
            self.ready = True
            self.workspace = tmp_path
            self.session_manager = sm
            self.api_channel = api_channel
            self.reliability_store = None
            self.agent_loop = None
            self.rag_service = None

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            sm.close()

        def status(self) -> dict:
            return {"ready": True}

    app = create_app(runtime_factory=_FakeRuntime)
    with TestClient(app) as client:
        # Create session & seed messages
        created = client.post("/v1/sessions")
        assert created.status_code == 201
        sid = created.json()["session"]["session_id"]

        sess = sm.get_or_create(f"api:{sid}")
        sess.add_message("user", "第一条提问")
        sess.add_message("assistant", "第一条回答", reasoning_content="内部推理")
        sess.add_message("user", "第二条提问")
        sess.add_message("assistant", "第二条回答")
        sm.save(sess)

        # Rename & pin session
        patched = client.patch(f"/v1/sessions/{sid}", json={"title": "重命名后的会话", "pinned": True})
        assert patched.status_code == 200
        assert patched.json()["title"] == "重命名后的会话"
        assert patched.json()["pinned"] is True

        # History includes seq and reasoning_content
        hist = client.get(f"/v1/sessions/{sid}/messages")
        assert hist.status_code == 200
        msgs = hist.json()["messages"]
        assert len(msgs) == 4
        assert msgs[1]["reasoning_content"] == "内部推理"
        second_user_seq = msgs[2]["seq"]

        # Truncate from second user message
        trunc = client.post(f"/v1/sessions/{sid}/truncate", json={"from_seq": second_user_seq})
        assert trunc.status_code == 200
        assert trunc.json()["deleted_count"] == 2

        hist_after = client.get(f"/v1/sessions/{sid}/messages").json()["messages"]
        assert len(hist_after) == 2

        # Upload image attachment & fetch preview
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        up = client.post(
            "/v1/chat/attachments",
            data={"session_id": sid},
            files={"file": ("demo.png", png_bytes, "image/png")},
        )
        assert up.status_code == 200
        att = up.json()
        assert att["is_image"] is True
        assert att["url"].startswith("/v1/chat/attachments/")

        fetched = client.get(att["url"])
        assert fetched.status_code == 200
        assert fetched.content == png_bytes
