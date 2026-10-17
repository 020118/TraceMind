""" Telegram 通道 —— 通过 Telegram Bot 与用户对话

架构：
    Telegram Bot (polling)
        │  OnMessage handler
        ▼
    publish_inbound(InboundMessage) → MessageBus
                                           │
                                           ▼ agent_loop
                                           │
    consume_outbound(OutboundMessage) ← MessageBus
        │  send_message to Telegram
        ▼
    Telegram 用户收到回复

配置（config.toml）：
    [channels.telegram]
    token = "xxx"              # Bot Token（必填）
    allow_from = ["username"]  # 允许的用户名列表（可选，留空=允许所有人）
"""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import Application, MessageHandler, filters

from agent.bus.events import InboundMessage, OutboundMessage
from agent.bus.queue import MessageBus

logger = logging.getLogger(__name__)


class TelegramChannel:
    """Telegram 通道 —— 管理 Bot 生命周期和消息路由"""

    def __init__(
        self,
        token: str,
        bus: MessageBus,
        allow_from: list[str] | None = None,
    ):
        self._token = token
        self._bus = bus
        self._allow_from = set(allow_from or [])
        self._app: Application | None = None

    async def start(self) -> None:
        """启动 Bot"""
        builder = Application.builder().token(self._token)
        self._app = builder.build()

        # 注册消息处理器
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

        # 开始 polling（非阻塞）
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(
            # Telegram 使用单数 "message"。保留 Bot 离线期间的积压更新，
            # 否则每次容器重启都会主动丢弃这些用户输入。
            allowed_updates=[Update.MESSAGE],
            drop_pending_updates=False,
        )
        logger.info("Telegram Bot 已启动")

    async def stop(self) -> None:
        """停止 Bot"""
        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                logger.warning("Telegram Bot 关闭异常: %s", e)
            self._app = None
        logger.info("Telegram Bot 已关闭")

    async def send(self, msg: OutboundMessage) -> None:
        """发送函数：把出站消息发到 Telegram"""
        if not self._app:
            logger.warning("Telegram Bot 未启动，无法发送消息")
            return
        if not msg.content or not msg.content.strip():
            logger.debug("跳过空消息: chat_id=%s", msg.chat_id)
            return
        # 失败必须向上抛给 dispatcher；如果在通道内吞掉异常，
        # dispatcher 会误以为投递成功，也就不可能进行重试。
        await self._app.bot.send_message(
            chat_id=int(msg.chat_id),
            text=msg.content,
            disable_web_page_preview=True,
        )

    # ── 入站：Telegram → MessageBus ───────────────────

    async def _on_message(self, update: Update, _context) -> None:
        """收到 Telegram 消息 → 发布到入站队列"""
        if not update.message or not update.message.text:
            return

        user = update.effective_user
        chat_id = str(update.effective_chat.id)
        username = user.username or user.full_name or "unknown"
        text = update.message.text.strip()

        # 权限检查
        if self._allow_from and username not in self._allow_from:
            logger.info("拒绝未授权用户: %s", username)
            await update.message.reply_text("抱歉，你没有权限使用此 Bot。")
            return

        logger.info("Telegram 消息: [%s] %s", username, text[:50])

        msg = InboundMessage(
            channel="telegram",
            sender_id=username,
            chat_id=chat_id,
            content=text,
        )
        await self._bus.publish_inbound(msg)


# ── 便捷启动函数 ───────────────────────────────────────

async def run_telegram_channel(
    token: str,
    bus: MessageBus,
    allow_from: list[str] | None = None,
) -> None:
    """启动 Telegram 通道（会阻塞直到 Bot 停止）"""
    channel = TelegramChannel(token, bus, allow_from)
    try:
        await channel.start()
        # 保持运行
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Telegram 通道被取消")
    finally:
        await channel.stop()
