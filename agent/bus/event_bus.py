import logging
import asyncio
import inspect
from typing import TypeAlias, TypeVar, cast
from collections.abc import Callable, Awaitable



logger = logging.getLogger(__name__)

E = TypeVar("E") # 定义泛型变量
Handler: TypeAlias = Callable[[E], Awaitable[E | None] | E | None]


class EventSubscription:
    """事件订阅句柄 —— 可用于取消注册 handler"""

    def __init__(self, bus: "EventBus", event_type: type[object], handler: Handler[object]) -> None:
        self._bus = bus
        self._event_type = event_type
        self._handler = handler

    def unsubscribe(self) -> None:
        """取消注册此 handler"""
        self._bus._remove_handler(self._event_type, self._handler)

    def __repr__(self) -> str:
        return f"EventSubscription(type={self._event_type.__name__}, handler={_handler_name(self._handler)})"


class EventBus:
    """Typed lifecycle hooks: observe + ordered intercept pipeline + fanout."""

    def __init__(self):
        self._handlers: dict[type[object], list[Handler[object]]] = {}
        self._any_handlers: list[Handler[object]] = []  # on_any 注册的通配 handler
        self._observe_queue: asyncio.Queue[object] | None = None
        self._observe_task: asyncio.Task[None] | None = None
        self._closed = False

    def on(self, event_type: type[E], handler: Handler[E]) -> EventSubscription:
        """注册类型化事件处理器，返回 EventSubscription 可用于取消"""
        handlers = self._handlers.setdefault(cast(type[object], event_type), [])
        raw_handler = cast(Handler[object], handler)
        handlers.append(raw_handler)
        return EventSubscription(self, cast(type[object], event_type), raw_handler)

    def on_any(self, handler: Handler[object]) -> EventSubscription:
        """注册通配处理器 —— 接收所有 emit/observe/fanout 事件

        适用于日志、监控等跨事件类型的场景。
        """
        self._any_handlers.append(handler)
        return EventSubscription(self, object, handler)  # object 作为通配标记

    def _remove_handler(self, event_type: type[object], handler: Handler[object]) -> None:
        """移除指定 handler（由 EventSubscription.unsubscribe 调用）"""
        if event_type is object:
            # on_any 注册的 handler
            try:
                self._any_handlers.remove(handler)
            except ValueError:
                pass
            return
        handlers = self._handlers.get(event_type)
        if handlers:
            try:
                handlers.remove(handler)
            except ValueError:
                pass
            if not handlers:
                self._handlers.pop(event_type, None)

    async def emit(self, event: E) -> E | None:
        """拦截链：依次执行，handler 返回 None 则终止，返回新事件则替换继续"""
        # 先跑通配 handler
        for h in self._any_handlers:
            result = h(event)
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                logger.info("emit 被通配 handler %s 拦截终止", _handler_name(h))
                return None
            event = cast(E, result)

        # 再跑类型化 handler
        for raw_handler in self._handlers.get(cast(type[object], type(event)), []):
            handler = cast(Handler[E], raw_handler)
            result = handler(event)
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                logger.info("emit 被 %s 拦截终止", _handler_name(handler))
                return None       # 拦截，事件被吃掉
            event = cast(E, result)  # 替换为 handler 返回的新事件
        return event

    async def observe(self, event: object) -> None:
        """观察者链：顺序通知，不修改事件，单个失败不打断"""
        for h in self._any_handlers:
            _ = await self._run_observe(event, h)
        for raw_handler in self._handlers.get(type(event), []):
            _ = await self._run_observe(event, raw_handler)

    async def fanout(self, event: object) -> None:
        """并发通知：同时执行所有处理器，汇总失败数"""
        # 通配 + 类型化
        all_handlers = list(self._any_handlers)
        all_handlers.extend(self._handlers.get(type(event), []))
        if not all_handlers:
            return
        results = await asyncio.gather(
            *(self._run_observe(event, h) for h in all_handlers)
        )
        fail_count = results.count(False)
        if fail_count:
            logger.warning(
                "fanout completed with observer errors: event=%s failed=%d total=%d",
                type(event).__name__,
                fail_count,
                len(all_handlers),
            )

    def enqueue(self, event: object) -> None:
        """异步入队：丢到后台队列，由 _run_observe_queue 执行 fanout"""
        if self._closed:
            logger.warning("event enqueue ignored after close: %s", type(event).__name__)
            return
        queue = self._ensure_observe_queue()
        queue.put_nowait(event)

    async def close(self) -> None:
        """关闭 EventBus：停止后台任务"""
        self._closed = True
        if self._observe_task is not None and not self._observe_task.done():
            self._observe_task.cancel()
            try:
                await self._observe_task
            except asyncio.CancelledError:
                pass
            self._observe_task = None

    def is_closed(self) -> bool:
        return self._closed

    # ── 内部方法 ──────────────────────────────────────

    def _ensure_observe_queue(self) -> asyncio.Queue[object]:
        if self._observe_queue is None:
            self._observe_queue = asyncio.Queue()
        self._ensure_observe_task()
        return self._observe_queue

    def _ensure_observe_task(self) -> None:
        if self._closed:
            return
        if self._observe_task is not None and not self._observe_task.done():
            return  # 已有运行中的任务
        task = asyncio.create_task(
            self._run_observe_queue(),
            name="event_bus_observe_queue"
        )
        self._observe_task = task
        task.add_done_callback(self._on_observe_task_done)

    def _on_observe_task_done(self, task: asyncio.Task[None]) -> None:
        if self._observe_task is task:
            self._observe_task = None
        if self._closed or task.cancelled():
            return
        try:
            exc = task.exception()
        except Exception as e:
            logger.warning("event dispatcher inspect failed: %s", e)
            exc = None
        if exc is not None:
            logger.error(
                "event dispatcher stopped unexpectedly: %s",
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        if self._observe_queue is not None:
            self._ensure_observe_task()

    async def _run_observe_queue(self) -> None:
        while True:
            queue = self._observe_queue
            if queue is None:
                return
            event = await queue.get()
            try:
                await self.fanout(event)
            finally:
                queue.task_done()

    async def _run_observe(self, event: object, handler: Handler[object]) -> bool:
        try:
            result = handler(event)
            if inspect.isawaitable(result):
                await result
            return True
        except Exception:
            logger.exception("observer error for %s handler=%s",
                             type(event).__name__,
                             _handler_name(handler))
            return False


def _handler_name(handler: Handler[object]) -> str:
    return str(
        getattr(
            handler,
            "__qualname__",
            getattr(handler, "__name__", repr(handler))
        )
    )
