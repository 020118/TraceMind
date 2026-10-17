""" 出站消息派发器 —— 解决多个通道竞争同一个出站队列的问题

问题：console_consumer 和 telegram._outbound_consumer 都从
      MessageBus.outbound 读取消息，但 asyncio.Queue 是点对点的，
      一条消息只能被一个消费者拿到。如果 console 抢走了 telegram 的消息，
      telegram 就永远收不到了。

解决：用一个派发器读取所有出站消息，按 channel 派发给对应的发送函数。、
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from agent.bus.events import OutboundMessage
from agent.bus.queue import MessageBus

logger = logging.getLogger(__name__)


# 发送函数类型：接收 OutboundMessage，返回 None
SendFn = Callable[[OutboundMessage], Awaitable[None]]


def _append_dead_letter(
    path: Path,
    msg: OutboundMessage,
    exc: Exception,
    attempts: int,
) -> None:
    """Persist one exhausted delivery without blocking the event loop."""

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "failed_at": datetime.now(UTC).isoformat(),
        "channel": msg.channel,
        "chat_id": msg.chat_id,
        "content": msg.content,
        "reply_to": msg.reply_to,
        "media": msg.media,
        "metadata": msg.metadata,
        "buttons": msg.buttons,
        "attempts": attempts,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


async def outbound_dispatcher(
    bus: MessageBus,
    handlers: dict[str, SendFn],
    *,
    max_attempts: int = 3,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 8.0,
    dead_letter_path: str | Path | None = None,
) -> None:
    """出站消息派发器

    从 bus.outbound 读取所有消息，按 msg.channel 派发到对应的 handler。

    参数:
        bus:      消息总线
        handlers: channel 名称 → 发送函数 的映射
                  如 {"console": console_send, "telegram": telegram_send}
    """
    while True:
        msg: OutboundMessage = await bus.consume_outbound()

        handler = handlers.get(msg.channel)
        if handler is None:
            logger.warning("未知的出站通道 '%s'，消息被丢弃: %s…", msg.channel, msg.content[:40])
            continue

        attempts = max(1, int(max_attempts))
        for attempt in range(1, attempts + 1):
            try:
                await handler(msg)
            except Exception as exc:
                try:
                    await bus.record_outbound_failure(msg, exc)
                except Exception:
                    logger.exception(
                        "出站失败状态落库失败 [channel=%s chat_id=%s]",
                        msg.channel,
                        msg.chat_id,
                    )
                if attempt >= attempts:
                    logger.error(
                        "出站消息发送失败，重试已耗尽 "
                        "[channel=%s chat_id=%s attempts=%d error_type=%s]: %s",
                        msg.channel,
                        msg.chat_id,
                        attempts,
                        type(exc).__name__,
                        exc,
                    )
                    if dead_letter_path is not None:
                        try:
                            await asyncio.to_thread(
                                _append_dead_letter,
                                Path(dead_letter_path),
                                msg,
                                exc,
                                attempts,
                            )
                        except Exception:
                            logger.exception(
                                "出站死信写入失败 [path=%s channel=%s chat_id=%s]",
                                dead_letter_path,
                                msg.channel,
                                msg.chat_id,
                            )
                    try:
                        await bus.mark_outbound_dead(msg, exc)
                    except Exception:
                        logger.exception(
                            "出站 DEAD 状态落库失败 [channel=%s chat_id=%s]",
                            msg.channel,
                            msg.chat_id,
                        )
                    break

                # Telegram RetryAfter 等异常会提供 retry_after；其他异常
                # 使用有上限的指数退避，避免网络抖动时立即连续重试。
                suggested_delay = getattr(exc, "retry_after", None)
                if suggested_delay is None:
                    delay = retry_base_delay * (2 ** (attempt - 1))
                else:
                    try:
                        delay = float(suggested_delay)
                    except (TypeError, ValueError):
                        delay = retry_base_delay * (2 ** (attempt - 1))
                delay = max(0.0, min(float(retry_max_delay), delay))
                logger.warning(
                    "出站消息发送失败，%.2fs 后重试 "
                    "[channel=%s chat_id=%s attempt=%d/%d error_type=%s]: %s",
                    delay,
                    msg.channel,
                    msg.chat_id,
                    attempt,
                    attempts,
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(delay)
            else:
                # 外部发送成功和本地 ACK 是两个不同结果。ACK 失败时不能
                # 立刻重发，否则会主动制造重复消息；保留 PENDING 供启动
                # 对账，语义仍是 at-least-once。
                try:
                    await bus.mark_outbound_sent(msg)
                except Exception:
                    logger.exception(
                        "出站消息已发送但 ACK 落库失败，保留待恢复状态 "
                        "[channel=%s chat_id=%s]",
                        msg.channel,
                        msg.chat_id,
                    )
                if attempt > 1:
                    logger.info(
                        "出站消息重试成功 [channel=%s attempt=%d/%d]",
                        msg.channel,
                        attempt,
                        attempts,
                    )
                break
