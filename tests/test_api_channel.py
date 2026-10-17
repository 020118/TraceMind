from __future__ import annotations

import asyncio

import pytest

from agent.bus.events import OutboundMessage
from agent.bus.queue import MessageBus
from agent.channels.api import ApiChannel


@pytest.mark.asyncio
async def test_api_channel_correlates_request_and_preserves_session() -> None:
    bus = MessageBus()
    channel = ApiChannel(bus)

    request_task = asyncio.create_task(
        channel.request(
            session_id="alice",
            sender_id=None,
            content="hello",
            timeout=1,
        )
    )
    inbound = await bus.consume_inbound()

    assert inbound.channel == "api"
    assert inbound.content == "hello"
    assert inbound.session_key == "api:alice"
    assert inbound.chat_id == inbound.metadata["request_id"]

    await channel.send(
        OutboundMessage(
            channel="api",
            chat_id=inbound.chat_id,
            content="world",
        )
    )
    request_id, outbound = await request_task

    assert request_id == inbound.chat_id
    assert outbound.content == "world"
    assert channel.pending_count == 0


@pytest.mark.asyncio
async def test_api_channel_cleans_pending_request_after_timeout() -> None:
    channel = ApiChannel(MessageBus())

    with pytest.raises(asyncio.TimeoutError):
        await channel.request(
            session_id="alice",
            sender_id=None,
            content="hello",
            timeout=0.01,
        )

    assert channel.pending_count == 0


@pytest.mark.asyncio
async def test_api_channel_streams_events_before_final_response() -> None:
    bus = MessageBus()
    channel = ApiChannel(bus)
    received: list[tuple[str, dict]] = []

    async def collect() -> None:
        async for item in channel.request_stream(
            session_id="alice",
            content="hello",
            timeout=1,
        ):
            received.append(item)

    task = asyncio.create_task(collect())
    inbound = await bus.consume_inbound()
    queue = inbound.metadata["_stream_event_queue"]
    await queue.put({"type": "answer_delta", "delta": "hel"})
    await channel.send(
        OutboundMessage(
            channel="api",
            chat_id=inbound.chat_id,
            content="hello",
        )
    )
    await task

    assert [event[1]["type"] for event in received] == [
        "answer_delta",
        "turn_finished",
    ]
    assert received[-1][1]["outbound"].content == "hello"
    assert channel.pending_count == 0
