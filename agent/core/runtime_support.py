"""运行期协议（迁移自 akashic agent.core.runtime_support，仅保留 proactive_v2 需要的 SessionLike）"""

from __future__ import annotations

from typing import Any, Protocol


class SessionLike(Protocol):
    key: str
    messages: list[dict]
    metadata: dict[str, object]
    last_consolidated: int

    def get_history(
        self,
        max_messages: int = 500,
        *,
        start_index: int | None = None,
    ) -> list[dict]: ...

    def add_message(self, role: str, content: str, media: Any = None, **kwargs: Any) -> None: ...
