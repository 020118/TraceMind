"""FastAPI channel adapter.

The HTTP layer publishes a normal :class:`InboundMessage` and waits for the
matching :class:`OutboundMessage`.  The agent core therefore remains unaware
of FastAPI and continues to use the same MessageBus as Console and Telegram.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from agent.bus.events import InboundMessage, OutboundMessage
from agent.bus.queue import MessageBus


class ApiChannel:
    """Bridge request/response HTTP calls onto the asynchronous MessageBus."""

    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self._pending: dict[str, asyncio.Future[OutboundMessage]] = {}
        self._session_locks: defaultdict[str, asyncio.Lock] = defaultdict(
            asyncio.Lock
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def request(
        self,
        *,
        session_id: str,
        content: str,
        sender_id: str | None = None,
        timeout: float = 120.0,
    ) -> tuple[str, OutboundMessage]:
        """Submit one turn and wait for its final outbound message.

        Requests for the same session are serialized so that conversation
        history cannot be observed out of order.  Different sessions may wait
        concurrently (the runtime can later scale its worker count without
        changing this channel contract).
        """

        request_id = uuid.uuid4().hex
        session_key = f"api:{session_id}"

        async with self._session_locks[session_key]:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[OutboundMessage] = loop.create_future()
            self._pending[request_id] = future

            inbound = InboundMessage(
                channel="api",
                sender_id=sender_id or session_id,
                chat_id=request_id,
                content=content,
                metadata={"request_id": request_id, "session_id": session_id},
                session_key_override=session_key,
            )
            await self._bus.publish_inbound(inbound)

            try:
                outbound = await asyncio.wait_for(future, timeout=timeout)
                return request_id, outbound
            finally:
                self._pending.pop(request_id, None)

    async def request_stream(
        self,
        *,
        session_id: str,
        content: str,
        sender_id: str | None = None,
        timeout: float = 120.0,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Submit one turn and yield structured progress plus the final reply.

        A per-request queue is attached only to the in-process inbound message.
        The Agent loop writes observable progress events to it while the normal
        Future remains the authoritative completion and persistence boundary.
        """

        request_id = uuid.uuid4().hex
        session_key = f"api:{session_id}"
        deadline = asyncio.get_running_loop().time() + timeout

        async with self._session_locks[session_key]:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[OutboundMessage] = loop.create_future()
            event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            self._pending[request_id] = future

            inbound = InboundMessage(
                channel="api",
                sender_id=sender_id or session_id,
                chat_id=request_id,
                content=content,
                metadata={
                    "request_id": request_id,
                    "session_id": session_id,
                    "_stream_event_queue": event_queue,
                },
                session_key_override=session_key,
            )
            await self._bus.publish_inbound(inbound)

            event_task: asyncio.Task[dict[str, Any]] | None = None
            try:
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError

                    event_task = asyncio.create_task(event_queue.get())
                    done, _ = await asyncio.wait(
                        {future, event_task},
                        timeout=remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        event_task.cancel()
                        raise asyncio.TimeoutError

                    if event_task in done:
                        yield request_id, event_task.result()
                    else:
                        event_task.cancel()

                    if future in done:
                        while not event_queue.empty():
                            yield request_id, event_queue.get_nowait()
                        yield request_id, {
                            "type": "turn_finished",
                            "outbound": future.result(),
                        }
                        return
            finally:
                if event_task is not None and not event_task.done():
                    event_task.cancel()
                self._pending.pop(request_id, None)

    async def send(self, message: OutboundMessage) -> None:
        """Resolve the HTTP request associated with an outbound message."""

        future = self._pending.get(message.chat_id)
        if future is not None and not future.done():
            future.set_result(message)

    async def close(self) -> None:
        """Cancel requests that are still waiting during application shutdown."""

        for future in list(self._pending.values()):
            if not future.done():
                future.cancel()
        self._pending.clear()

