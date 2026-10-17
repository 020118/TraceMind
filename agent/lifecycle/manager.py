""" 生命周期管理器 —— 管理应用阶段转换

阶段流：STARTING → READY → STOPPING → STOPPED
每个阶段转换会通过 EventBus 发布 LifecycleEvent。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agent.bus.event_bus import EventBus
from agent.bus.events import LifecycleEvent, LifecyclePhase

logger = logging.getLogger(__name__)

# 阶段转换回调签名
PhaseCallback = Callable[[], Awaitable[None] | None]


class LifecycleManager:
    """应用生命周期管理器"""

    def __init__(self, events: EventBus):
        self._events = events
        self._phase = LifecyclePhase.STOPPED
        self._callbacks: dict[LifecyclePhase, list[PhaseCallback]] = {
            p: [] for p in LifecyclePhase
        }

    @property
    def phase(self) -> LifecyclePhase:
        """当前生命周期阶段"""
        return self._phase

    @property
    def is_running(self) -> bool:
        """是否处于运行中（STARTING 或 READY）"""
        return self._phase in (LifecyclePhase.STARTING, LifecyclePhase.READY)

    def on_phase(self, phase: LifecyclePhase, callback: PhaseCallback) -> None:
        """注册阶段转换回调

        当进入 phase 阶段时会依次执行所有已注册的回调。
        """
        self._callbacks[phase].append(callback)

    def on_starting(self, callback: PhaseCallback) -> None:
        """注册 STARTING 阶段回调"""
        self.on_phase(LifecyclePhase.STARTING, callback)

    def on_ready(self, callback: PhaseCallback) -> None:
        """注册 READY 阶段回调"""
        self.on_phase(LifecyclePhase.READY, callback)

    def on_stopping(self, callback: PhaseCallback) -> None:
        """注册 STOPPING 阶段回调"""
        self.on_phase(LifecyclePhase.STOPPING, callback)

    async def start(self) -> None:
        """启动：STARTING → READY

        1. 设置阶段为 STARTING
        2. 执行 STARTING 回调
        3. 通过 EventBus 发布 LifecycleEvent(STARTING)
        4. 设置阶段为 READY
        5. 执行 READY 回调
        6. 发布 LifecycleEvent(READY)
        """
        if self._phase != LifecyclePhase.STOPPED:
            logger.warning("寿命周期已在 %s 状态，忽略 start()", self._phase.value)
            return

        logger.info("生命周期: STARTING")
        self._phase = LifecyclePhase.STARTING
        await self._run_callbacks(LifecyclePhase.STARTING, "应用启动中...")

        logger.info("生命周期: READY")
        self._phase = LifecyclePhase.READY
        await self._run_callbacks(LifecyclePhase.READY, "应用已就绪")

    async def stop(self) -> None:
        """停止：STOPPING → STOPPED"""
        if self._phase != LifecyclePhase.READY:
            logger.warning("生命周期不在 READY 状态，忽略 stop()")
            return

        logger.info("生命周期: STOPPING")
        self._phase = LifecyclePhase.STOPPING
        await self._run_callbacks(LifecyclePhase.STOPPING, "应用停止中...")

        logger.info("生命周期: STOPPED")
        self._phase = LifecyclePhase.STOPPED
        await self._run_callbacks(LifecyclePhase.STOPPED, "应用已停止")

    async def _run_callbacks(self, phase: LifecyclePhase, message: str) -> None:
        """执行阶段回调 + 发布 EventBus 事件"""
        # 先执行回调
        for cb in self._callbacks.get(phase, []):
            try:
                result = cb()
                if result is not None:
                    await result
            except Exception:
                logger.exception("生命周期回调异常: phase=%s", phase.value)

        # 再发布事件
        try:
            await self._events.observe(LifecycleEvent(phase=phase, message=message))
        except Exception:
            logger.exception("生命周期事件发布异常: phase=%s", phase.value)
