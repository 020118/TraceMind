from __future__ import annotations

import pytest

from agent.bus.events import OutboundMessage
from agent.bus.queue import MessageBus
from agent.channels.telegram import TelegramChannel


class _FailingBot:
    async def send_message(self, **_kwargs) -> None:
        raise TimeoutError("network jitter")


class _FakeApplication:
    bot = _FailingBot()


@pytest.mark.asyncio
async def test_send_propagates_failure_to_dispatcher() -> None:
    """Telegram 通道不能吞掉错误，否则 dispatcher 无法重试。"""

    channel = TelegramChannel("test-token", MessageBus())
    channel._app = _FakeApplication()  # type: ignore[assignment]

    with pytest.raises(TimeoutError, match="network jitter"):
        await channel.send(
            OutboundMessage(
                channel="telegram",
                chat_id="123",
                content="answer",
            )
        )
