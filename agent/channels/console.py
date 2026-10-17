""" 控制台通道 —— 标准输入输出通道

既作为 Inbound 生产者（读取用户键盘输入），
也作为 Outbound 消费者（打印 AI 回复到屏幕）。
"""

from __future__ import annotations

import asyncio

from agent.bus.events import InboundMessage, OutboundMessage
from agent.bus.queue import MessageBus


async def console_producer(bus: MessageBus) -> None:
    """生产者：从标准输入读取用户消息，推入入站队列"""
    loop = asyncio.get_event_loop()
    print("=" * 40)
    print("AI 助手已启动，输入你的消息（输入 exit 退出）：")
    print("=" * 40)

    while True:
        text = await loop.run_in_executor(None, input, "\n你: ")
        if text.strip().lower() == "exit":
            print("再见！")
            break

        msg = InboundMessage(
            channel="console",
            sender_id="user",
            chat_id="console",
            content=text,
        )
        await bus.publish_inbound(msg)


async def console_send(msg: OutboundMessage) -> None:
    """发送函数：把出站消息打印到控制台"""
    print(f"\nAI: {msg.content}")
    if msg.buttons:
        for row_idx, row in enumerate(msg.buttons):
            print(f"  按钮行 {row_idx + 1}: {' | '.join(row)}")


async def run_console_channel(bus: MessageBus) -> None:
    """启动控制台通道（仅生产者，消费者由 dispatcher 管理）"""
    await console_producer(bus)
