""" 异步消息队列，用于解耦通道和 Agent 之间的通信 """

import asyncio
from typing import TYPE_CHECKING

from agent.bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    from agent.reliability import ReliabilityStore


class MessageBus:
    """异步消息总线，解耦聊天通道与 Agent 核心

    通道生产者将消息推入 inbound 队列 → Agent 消费者从中获取进行处理
    Agent 生产者将回复推入 outbound 队列 → 通道消费者从中获取并发送

    双队列 + async 非阻塞设计：通道和 Agent 可完全独立运行，互不阻塞。
    """

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._outbox_store: ReliabilityStore | None = None
        self._durable_channels: set[str] = set()

    def configure_outbox(
        self,
        store: "ReliabilityStore",
        *,
        durable_channels: set[str] | None = None,
    ) -> None:
        """Persist selected channel messages before they enter the memory queue."""

        self._outbox_store = store
        self._durable_channels = set(durable_channels or {"telegram"})

    # ── 入站：通道 → Agent ─────────────────────────────

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """通道推送消息给 Agent"""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Agent 获取下一条入站消息（阻塞直到有消息）"""
        return await self.inbound.get()

    # ── 出站：Agent → 通道 ─────────────────────────────

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Agent 推送回复给通道"""
        if (
            self._outbox_store is not None
            and msg.channel in self._durable_channels
            and not msg.metadata.get("_outbox_id")
        ):
            payload = {
                "channel": msg.channel,
                "chat_id": msg.chat_id,
                "content": msg.content,
                "reply_to": msg.reply_to,
                "media": list(msg.media),
                "metadata": dict(msg.metadata),
                "buttons": [list(row) for row in msg.buttons],
            }
            outbox_id = await asyncio.to_thread(
                self._outbox_store.enqueue_outbound, payload
            )
            msg.metadata["_outbox_id"] = outbox_id
        await self.outbound.put(msg)

    async def restore_pending_outbound(self) -> int:
        """Replay durable messages without inserting duplicate outbox rows."""

        if self._outbox_store is None:
            return 0
        records = await asyncio.to_thread(self._outbox_store.list_pending_outbound)
        for record in records:
            metadata = dict(record.get("metadata") or {})
            metadata["_outbox_id"] = record["outbox_id"]
            await self.outbound.put(
                OutboundMessage(
                    channel=str(record.get("channel", "")),
                    chat_id=str(record.get("chat_id", "")),
                    content=str(record.get("content", "")),
                    reply_to=record.get("reply_to"),
                    media=list(record.get("media") or []),
                    metadata=metadata,
                    buttons=[list(row) for row in (record.get("buttons") or [])],
                )
            )
        return len(records)

    async def mark_outbound_sent(self, msg: OutboundMessage) -> None:
        outbox_id = str(msg.metadata.get("_outbox_id", ""))
        if self._outbox_store is not None and outbox_id:
            await asyncio.to_thread(self._outbox_store.mark_outbound_sent, outbox_id)

    async def record_outbound_failure(
        self, msg: OutboundMessage, exc: Exception
    ) -> None:
        outbox_id = str(msg.metadata.get("_outbox_id", ""))
        if self._outbox_store is not None and outbox_id:
            await asyncio.to_thread(
                self._outbox_store.record_outbound_failure,
                outbox_id,
                str(exc),
            )

    async def mark_outbound_dead(
        self, msg: OutboundMessage, exc: Exception
    ) -> None:
        outbox_id = str(msg.metadata.get("_outbox_id", ""))
        if self._outbox_store is not None and outbox_id:
            await asyncio.to_thread(
                self._outbox_store.mark_outbound_dead,
                outbox_id,
                str(exc),
            )

    async def consume_outbound(self) -> OutboundMessage:
        """通道获取下一条出站消息（阻塞直到有消息）"""
        return await self.outbound.get()

    # ── 队列状态 ───────────────────────────────────────

    @property
    def inbound_size(self) -> int:
        """待处理入站消息数量"""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """待处理出站消息数量"""
        return self.outbound.qsize()
