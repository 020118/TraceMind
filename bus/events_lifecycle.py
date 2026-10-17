"""生命周期事件（迁移自 akashic bus.events_lifecycle，仅保留 proactive_v2 需要的两个事件）"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class ProactiveFinished:
    session_key: str
    tick_id: str
    mode: Literal["proactive", "drift"]
    terminal_action: str | None
    gate_exit: str | None
    skip_reason: str
    steps_taken: int
    alert_count: int
    content_count: int
    context_count: int
    final_message: str
    llm_call_count: int
    cache_prompt_tokens: int | None = None
    cache_hit_tokens: int | None = None
    timestamp: datetime | None = None


@dataclass(frozen=True)
class DriftFinished:
    session_key: str
    skill_name: str
    status: str
    briefing: str
    message_result: str
    timestamp: datetime
