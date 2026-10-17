""" TurnLifecycle 钩子链测试

运行方式：
    pytest tests/test_lifecycle_turn.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from agent.bus.event_bus import EventBus
from agent.bus.events import InboundMessage
from agent.lifecycle.turn import TurnLifecycle
from agent.lifecycle.types import (
    AfterReasoningCtx,
    AfterStepCtx,
    AfterTurnCtx,
    BeforeStepCtx,
    BeforeTurnCtx,
    PromptRenderCtx,
)


def _msg(content: str = "你好") -> InboundMessage:
    return InboundMessage(channel="console", sender_id="u1", chat_id="c1", content=content)


# ── before_turn 门 ──────────────────────────────────────


async def test_before_turn_passthrough():
    bus = EventBus()
    lc = TurnLifecycle(bus)
    bt = await lc.run_before_turn(_msg(), memory_block="记忆块")
    assert bt.abort is False
    assert bt.content == "你好"
    assert bt.memory_block == "记忆块"
    assert bt.session_key == "console:c1"


async def test_before_turn_handler_returns_none_aborts():
    bus = EventBus()

    async def block(ctx: BeforeTurnCtx):
        return None

    bus.on(BeforeTurnCtx, block)
    lc = TurnLifecycle(bus)
    bt = await lc.run_before_turn(_msg())
    assert bt.abort is True


async def test_before_turn_handler_can_abort_with_reply():
    bus = EventBus()

    async def block(ctx: BeforeTurnCtx):
        ctx.abort = True
        ctx.abort_reply = "不能发这个"
        return ctx

    bus.on(BeforeTurnCtx, block)
    lc = TurnLifecycle(bus)
    bt = await lc.run_before_turn(_msg())
    assert bt.abort is True
    assert bt.abort_reply == "不能发这个"


# ── prompt_render 门 ────────────────────────────────────


async def test_prompt_render_assembles_messages():
    bus = EventBus()
    lc = TurnLifecycle(bus)
    messages = await lc.run_prompt_render(
        channel="console",
        chat_id="c1",
        content="hello",
        system_parts=["你是助手"],
        history=[{"role": "user", "content": "上一轮"}],
        memory_block="记忆",
    )
    assert messages[0]["role"] == "system"
    assert "你是助手" in messages[0]["content"]
    assert "记忆" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "hello"}
    assert len(messages) == 3  # system + 1 条历史 + 当前消息


async def test_prompt_render_handler_can_append_system_parts():
    bus = EventBus()

    async def add_rule(ctx: PromptRenderCtx):
        ctx.system_parts.append("额外规则")
        return ctx

    bus.on(PromptRenderCtx, add_rule)
    lc = TurnLifecycle(bus)
    messages = await lc.run_prompt_render(
        channel="console", chat_id="c1", content="hi",
        system_parts=["你是助手"], history=[],
    )
    assert "额外规则" in messages[0]["content"]


# ── before_step 门 ──────────────────────────────────────


async def test_before_step_passthrough():
    bus = EventBus()
    lc = TurnLifecycle(bus)
    bs = await lc.run_before_step(
        channel="console", chat_id="c1", iteration=0, messages=[], tool_schemas=None,
    )
    assert bs.early_stop is False
    assert bs.iteration == 0


async def test_before_step_handler_can_early_stop():
    bus = EventBus()

    async def stop(ctx: BeforeStepCtx):
        ctx.early_stop = True
        ctx.early_stop_reply = "够了"
        return ctx

    bus.on(BeforeStepCtx, stop)
    lc = TurnLifecycle(bus)
    bs = await lc.run_before_step(
        channel="console", chat_id="c1", iteration=3, messages=[], tool_schemas=None,
    )
    assert bs.early_stop is True
    assert bs.early_stop_reply == "够了"


# ── after_reasoning 门 ──────────────────────────────────


async def test_after_reasoning_can_rewrite_reply():
    bus = EventBus()

    async def rewrite(ctx: AfterReasoningCtx):
        ctx.reply = "改写:" + ctx.reply
        return ctx

    bus.on(AfterReasoningCtx, rewrite)
    lc = TurnLifecycle(bus)
    ar = await lc.run_after_reasoning(
        channel="console", chat_id="c1", content="q", reply="原始回复", tools_used=(),
    )
    assert ar.reply == "改写:原始回复"


# ── after_turn / after_step 观察 ────────────────────────


async def test_after_turn_observe_called():
    bus = EventBus()
    seen: list[str] = []

    async def tap(ctx: AfterTurnCtx):
        seen.append(ctx.reply)

    bus.on(AfterTurnCtx, tap)
    lc = TurnLifecycle(bus)
    await lc.run_after_turn(
        channel="console", chat_id="c1", content="q", reply="r", tools_used=("echo",),
    )
    assert seen == ["r"]


async def test_after_step_observe_called():
    bus = EventBus()
    seen: list[tuple[str, ...]] = []

    async def tap(ctx: AfterStepCtx):
        seen.append(ctx.tool_names)

    bus.on(AfterStepCtx, tap)
    lc = TurnLifecycle(bus)
    await lc.run_after_step(
        channel="console", chat_id="c1", iteration=1,
        tool_names=("echo", "weather"), has_more=True,
    )
    assert seen == [("echo", "weather")]


# ── inspect 调试 ────────────────────────────────────────


def test_inspect_lists_all_phases():
    lc = TurnLifecycle(EventBus())
    text = lc.inspect()
    for name in ("before_turn", "before_reasoning", "prompt_render", "before_step",
                 "after_step", "after_reasoning", "after_turn"):
        assert name in text
