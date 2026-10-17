""" outbound_dispatcher 测试

运行方式：
    pytest tests/test_dispatcher.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from agent.channels.dispatcher import outbound_dispatcher
from agent.bus.events import OutboundMessage
from agent.bus.queue import MessageBus


async def start_dispatcher(
    handlers: dict,
    **dispatcher_kwargs,
) -> tuple[MessageBus, asyncio.Task]:
    """启动派发器后台任务，返回 (bus, task)"""
    bus = MessageBus()
    task = asyncio.create_task(
        outbound_dispatcher(bus, handlers, **dispatcher_kwargs)
    )
    return bus, task


async def make_message(channel: str, content: str = "test") -> OutboundMessage:
    return OutboundMessage(channel=channel, chat_id="chat1", content=content)


# ── 正确路由 ────────────────────────────────────────────

async def test_routes_to_matching_handler():
    """console 通道的消息被派发到 console handler"""
    received = asyncio.Event()
    captured = []

    async def fake_console_send(msg):
        captured.append(msg)
        received.set()

    bus, task = await start_dispatcher({"console": fake_console_send})

    await bus.publish_outbound(await make_message("console"))
    await asyncio.wait_for(received.wait(), timeout=1)

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert len(captured) == 1
    assert captured[0].channel == "console"


async def test_routes_to_correct_handler():
    """不同通道的消息被派发到各自的 handler"""
    # TODO: 补全这个测试
    # 提示：注册两个 handler，各用一个 Event，发两条不同 channel 的消息
    console_received = asyncio.Event()
    telegram_received = asyncio.Event()

    async def fake_console_send(msg):
        console_received.set()
    
    async def fake_telegram_send(msg):
        telegram_received.set()
    
    bus, task = await start_dispatcher({"console": fake_console_send, "telegram": fake_telegram_send})
    await bus.publish_outbound(await make_message("console"))
    await bus.publish_outbound(await make_message("telegram"))

    await asyncio.wait_for(console_received.wait(), timeout=1)
    await asyncio.wait_for(telegram_received.wait(), timeout=1)

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)



# ── 未知通道 ────────────────────────────────────────────

async def test_unknown_channel_dropped():
    """没有 handler 的通道：消息被丢弃但不崩溃"""
    # TODO: 补全这个测试
    # 提示：只注册 console handler，发一条 telegram 消息，程序不应崩溃
    console_received = asyncio.Event()

    async def fake_console_send(msg):
        console_received.set()
    
    bus, task = await start_dispatcher({"console" : fake_console_send})
    await bus.publish_outbound(await make_message("telegram"))
    await bus.publish_outbound(await make_message("console"))
    await asyncio.wait_for(console_received.wait(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


# ── handler 异常 ────────────────────────────────────────

async def test_handler_error_does_not_stop():
    """handler 抛异常不影响后续消息"""
    # TODO: 补全这个测试
    # 提示：第一个 handler 抛异常，第二条消息仍能被派发
    console_received = asyncio.Event()

    async def fake_console_send(msg):
        console_received.set()

    async def bad_console_send(msg):
        raise RuntimeError("boom")

    bus, task = await start_dispatcher(
        {"console": fake_console_send, "bad_console": bad_console_send},
        retry_base_delay=0,
    )
    await bus.publish_outbound(await make_message("bad_console"))
    await bus.publish_outbound(await make_message("console"))
    await asyncio.wait_for(console_received.wait(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_retries_transient_handler_failure():
    """发送端短暂失败时重试同一条消息，成功后停止重试。"""
    delivered = asyncio.Event()
    attempts = 0

    async def flaky_send(_msg):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary network failure")
        delivered.set()

    bus, task = await start_dispatcher(
        {"telegram": flaky_send},
        max_attempts=3,
        retry_base_delay=0,
    )
    await bus.publish_outbound(await make_message("telegram"))
    await asyncio.wait_for(delivered.wait(), timeout=1)

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert attempts == 3


async def test_ack_persistence_failure_does_not_resend_delivered_message():
    """外部发送成功后，本地 ACK 失败不能立刻再次调用发送端。"""
    delivered = asyncio.Event()
    attempts = 0
    bus = MessageBus()

    async def succeeds(_msg):
        nonlocal attempts
        attempts += 1
        delivered.set()

    async def fail_ack(_msg):
        raise OSError("injected ack database failure")

    bus.mark_outbound_sent = fail_ack  # type: ignore[method-assign]
    task = asyncio.create_task(
        outbound_dispatcher(
            bus,
            {"telegram": succeeds},
            max_attempts=3,
            retry_base_delay=0,
        )
    )
    await bus.publish_outbound(await make_message("telegram"))
    await asyncio.wait_for(delivered.wait(), timeout=1)
    await asyncio.sleep(0.02)

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert attempts == 1


async def test_retry_exhaustion_does_not_block_later_messages():
    """一条消息重试耗尽后，dispatcher 仍继续处理后续消息。"""
    later_delivered = asyncio.Event()
    bad_attempts = 0

    async def always_fails(_msg):
        nonlocal bad_attempts
        bad_attempts += 1
        raise TimeoutError("still unavailable")

    async def succeeds(_msg):
        later_delivered.set()

    bus, task = await start_dispatcher(
        {"telegram": always_fails, "console": succeeds},
        max_attempts=3,
        retry_base_delay=0,
    )
    await bus.publish_outbound(await make_message("telegram"))
    await bus.publish_outbound(await make_message("console"))
    await asyncio.wait_for(later_delivered.wait(), timeout=1)

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert bad_attempts == 3


async def test_retry_exhaustion_persists_dead_letter(tmp_path):
    """重试耗尽后保留消息与错误上下文，不静默丢失回复。"""
    later_delivered = asyncio.Event()
    dead_letter_path = tmp_path / "dead_letter" / "outbound.jsonl"

    async def always_fails(_msg):
        raise TimeoutError("telegram unavailable")

    async def succeeds(_msg):
        later_delivered.set()

    bus, task = await start_dispatcher(
        {"telegram": always_fails, "console": succeeds},
        max_attempts=2,
        retry_base_delay=0,
        dead_letter_path=dead_letter_path,
    )
    await bus.publish_outbound(await make_message("telegram", "important answer"))
    await bus.publish_outbound(await make_message("console"))
    await asyncio.wait_for(later_delivered.wait(), timeout=1)

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    records = [
        json.loads(line)
        for line in dead_letter_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["channel"] == "telegram"
    assert records[0]["content"] == "important answer"
    assert records[0]["attempts"] == 2
    assert records[0]["error_type"] == "TimeoutError"
