""" 生命周期 Phase 钩子链的上下文类型

每个 turn（一轮对话）被拆成 7 个阶段，各自有一个上下文类型：

    before_turn       门（gate）：可拦截/中止，注入提示
    before_reasoning  门（gate）：可中止
    prompt_render     门（gate）：可增删 system prompt 片段
    before_step       门（gate）：每次 LLM 调用前，可提前终止
    after_step        观察（tap）：单次推理步后快照
    after_reasoning   门（gate）：可改写最终回复
    after_turn        观察（tap）：回复即将派发前的快照

约定：带 "writable" 注释的字段是模块/处理器可改写的；其余为只读快照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _empty_str_list() -> list[str]:
    return []


@dataclass
class BeforeTurnCtx:
    """before_turn 上下文：消息刚进入循环，尚未推理。"""

    channel: str
    sender_id: str
    chat_id: str
    content: str
    timestamp: datetime
    session_key: str
    media: list[str] = field(default_factory=_empty_str_list)
    memory_block: str = ""
    # writable
    abort: bool = False
    abort_reply: str = ""


@dataclass
class BeforeReasoningCtx:
    """before_reasoning 上下文：准备开始推理前的最后一道路障。"""

    channel: str
    sender_id: str
    chat_id: str
    content: str
    session_key: str
    memory_block: str = ""
    # writable
    abort: bool = False
    abort_reply: str = ""


@dataclass
class PromptRenderCtx:
    """prompt_render 上下文：允许在渲染前增删 system prompt 片段。"""

    channel: str
    chat_id: str
    content: str
    system_parts: list[str] = field(default_factory=_empty_str_list)
    history: list[dict[str, Any]] = field(default_factory=list)
    memory_block: str = ""


@dataclass
class BeforeStepCtx:
    """before_step 上下文：单次 LLM 调用前，可提前终止工具循环。"""

    channel: str
    chat_id: str
    iteration: int
    messages: list[dict[str, Any]]
    tool_schemas: list[dict] | None
    # writable
    early_stop: bool = False
    early_stop_reply: str = ""


@dataclass
class AfterStepCtx:
    """after_step 上下文：单次推理步后的只读快照（tap）。"""

    channel: str
    chat_id: str
    iteration: int
    tool_names: tuple[str, ...]
    has_more: bool


@dataclass
class AfterReasoningCtx:
    """after_reasoning 上下文：可改写最终回复文本。"""

    channel: str
    chat_id: str
    content: str
    tools_used: tuple[str, ...]
    # writable
    reply: str = ""


@dataclass
class AfterTurnCtx:
    """after_turn 上下文：回复即将派发前的只读快照（tap）。"""

    channel: str
    chat_id: str
    content: str
    reply: str
    tools_used: tuple[str, ...]
    will_dispatch: bool = True
