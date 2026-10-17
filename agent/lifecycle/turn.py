""" TurnLifecycle —— 一轮对话的 7 阶段钩子链门面

把 agent_loop 的处理过程拆成 7 个 Phase，每个 Phase 内部按依赖顺序编排内置模块，
并把 EventBus 作为扩展点（emit=门/可改写，observe=观察）。

阶段流：
    before_turn(门) → before_reasoning(门) → prompt_render(门)
    → [before_step(门) → (LLM) → after_step(观察)] × N
    → after_reasoning(门) → after_turn(观察)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from agent.bus.event_bus import EventBus
from agent.bus.events import InboundMessage
from agent.lifecycle.phase import Phase, PhaseFrame, topo_sort_modules
from agent.lifecycle.types import (
    AfterReasoningCtx,
    AfterStepCtx,
    AfterTurnCtx,
    BeforeReasoningCtx,
    BeforeStepCtx,
    BeforeTurnCtx,
    PromptRenderCtx,
)


@dataclass
class TurnFrame(PhaseFrame[Any, Any]):
    """所有 phase 共用的 frame：input → modules → output。"""


# ── Phase 输入类型（内部管线数据） ────────────────────────


@dataclass
class BeforeTurnInput:
    msg: InboundMessage
    memory_block: str = ""


@dataclass
class PromptRenderInput:
    channel: str
    chat_id: str
    content: str
    system_parts: list[str]
    history: list[dict[str, Any]]
    memory_block: str = ""


@dataclass
class BeforeStepInput:
    channel: str
    chat_id: str
    iteration: int
    messages: list[dict[str, Any]]
    tool_schemas: list[dict] | None


@dataclass
class AfterStepInput:
    channel: str
    chat_id: str
    iteration: int
    tool_names: tuple[str, ...]
    has_more: bool


@dataclass
class AfterReasoningInput:
    channel: str
    chat_id: str
    content: str
    reply: str
    tools_used: tuple[str, ...]


@dataclass
class AfterTurnInput:
    channel: str
    chat_id: str
    content: str
    reply: str
    tools_used: tuple[str, ...]
    will_dispatch: bool = True


# ── ctx 构建函数 ─────────────────────────────────────────


def _build_before_turn(inp: BeforeTurnInput) -> BeforeTurnCtx:
    msg = inp.msg
    return BeforeTurnCtx(
        channel=msg.channel,
        sender_id=msg.sender_id,
        chat_id=msg.chat_id,
        content=msg.content,
        timestamp=msg.timestamp,
        session_key=msg.session_key,
        media=list(msg.media),
        memory_block=inp.memory_block,
    )


def _build_before_reasoning(inp: BeforeTurnCtx) -> BeforeReasoningCtx:
    return BeforeReasoningCtx(
        channel=inp.channel,
        sender_id=inp.sender_id,
        chat_id=inp.chat_id,
        content=inp.content,
        session_key=inp.session_key,
        memory_block=inp.memory_block,
    )


def _build_prompt_render(inp: PromptRenderInput) -> PromptRenderCtx:
    return PromptRenderCtx(
        channel=inp.channel,
        chat_id=inp.chat_id,
        content=inp.content,
        system_parts=list(inp.system_parts),
        history=list(inp.history),
        memory_block=inp.memory_block,
    )


def _build_before_step(inp: BeforeStepInput) -> BeforeStepCtx:
    return BeforeStepCtx(
        channel=inp.channel,
        chat_id=inp.chat_id,
        iteration=inp.iteration,
        messages=inp.messages,
        tool_schemas=inp.tool_schemas,
    )


def _build_after_step(inp: AfterStepInput) -> AfterStepCtx:
    return AfterStepCtx(
        channel=inp.channel,
        chat_id=inp.chat_id,
        iteration=inp.iteration,
        tool_names=inp.tool_names,
        has_more=inp.has_more,
    )


def _build_after_reasoning(inp: AfterReasoningInput) -> AfterReasoningCtx:
    return AfterReasoningCtx(
        channel=inp.channel,
        chat_id=inp.chat_id,
        content=inp.content,
        tools_used=inp.tools_used,
        reply=inp.reply,
    )


def _build_after_turn(inp: AfterTurnInput) -> AfterTurnCtx:
    return AfterTurnCtx(
        channel=inp.channel,
        chat_id=inp.chat_id,
        content=inp.content,
        reply=inp.reply,
        tools_used=inp.tools_used,
        will_dispatch=inp.will_dispatch,
    )


# ── 通用模块 ─────────────────────────────────────────────


class _Build:
    """构建 ctx 并写入 produces[0] slot。"""

    def __init__(self, builder, *, slot: str, produces: tuple[str, ...]) -> None:
        self.slot = slot
        self.requires: tuple[str, ...] = ()
        self.produces = produces
        self._builder = builder

    async def run(self, frame: TurnFrame) -> TurnFrame:
        frame.slots[self.produces[0]] = self._builder(frame.input)
        return frame


class _Emit:
    """门（gate）：把 ctx 交给 EventBus.emit，处理器可改写或拦截。

    requires 是模块依赖（用于拓扑排序）；data_slot 是实际读取/写入的数据槽。
    """

    def __init__(self, bus: EventBus, *, slot: str, requires: tuple[str, ...], data_slot: str) -> None:
        self.slot = slot
        self.requires = requires
        self.produces = (data_slot,)
        self._bus = bus
        self._data_slot = data_slot

    async def run(self, frame: TurnFrame) -> TurnFrame:
        prev = frame.slots[self._data_slot]
        result = await self._bus.emit(prev)
        if result is None:
            # 兼容"返回 None 即拦截"的旧式 handler：转为 abort/early_stop 标志
            for attr in ("abort", "early_stop"):
                if hasattr(prev, attr):
                    setattr(prev, attr, True)
                    break
            result = prev
        frame.slots[self._data_slot] = result
        return frame


class _Observe:
    """观察（tap）：把 ctx 交给 EventBus.observe，不修改数据。"""

    def __init__(self, bus: EventBus, *, slot: str, requires: tuple[str, ...], data_slot: str) -> None:
        self.slot = slot
        self.requires = requires
        self.produces: tuple[str, ...] = ()
        self._bus = bus
        self._data_slot = data_slot

    async def run(self, frame: TurnFrame) -> TurnFrame:
        await self._bus.observe(frame.slots[self._data_slot])
        return frame


class _RenderMessages:
    """把 system_parts + memory_block + history + 当前消息组装成 messages。"""

    def __init__(self, *, slot: str, requires: tuple[str, ...], data_slot: str, result_slot: str) -> None:
        self.slot = slot
        self.requires = requires
        self.produces = (result_slot,)
        self._data_slot = data_slot
        self._result_slot = result_slot

    async def run(self, frame: TurnFrame) -> TurnFrame:
        ctx = cast(PromptRenderCtx, frame.slots[self._data_slot])
        parts = list(ctx.system_parts)
        if ctx.memory_block:
            parts.append("")
            parts.append(ctx.memory_block)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "\n".join(parts)}
        ]
        messages.extend(ctx.history)
        messages.append({"role": "user", "content": ctx.content})
        frame.slots[self._result_slot] = messages
        return frame


class _Return:
    """把 data_slot 作为 phase 的 output。"""

    def __init__(self, *, slot: str, requires: tuple[str, ...], data_slot: str) -> None:
        self.slot = slot
        self.requires = requires
        self.produces: tuple[str, ...] = ()
        self._data_slot = data_slot

    async def run(self, frame: TurnFrame) -> TurnFrame:
        frame.output = frame.slots[self._data_slot]
        return frame


# ── 模块工厂 ─────────────────────────────────────────────


def _gate_modules(phase: str, bus: EventBus, builder) -> list[object]:
    """门阶段：build → emit → return。"""
    ctx = f"{phase}:ctx"
    build = f"{phase}.build"
    emit = f"{phase}.emit"
    ret = f"{phase}.return"
    return topo_sort_modules([
        _Build(builder, slot=build, produces=(ctx,)),
        _Emit(bus, slot=emit, requires=(build,), data_slot=ctx),
        _Return(slot=ret, requires=(emit,), data_slot=ctx),
    ])


def _tap_modules(phase: str, bus: EventBus, builder) -> list[object]:
    """观察阶段：build → observe → return。"""
    ctx = f"{phase}:ctx"
    build = f"{phase}.build"
    obs = f"{phase}.observe"
    ret = f"{phase}.return"
    return topo_sort_modules([
        _Build(builder, slot=build, produces=(ctx,)),
        _Observe(bus, slot=obs, requires=(build,), data_slot=ctx),
        _Return(slot=ret, requires=(obs,), data_slot=ctx),
    ])


def _prompt_render_modules(bus: EventBus) -> list[object]:
    """渲染阶段：build → emit（可改 system_parts）→ render → return。"""
    ctx = "prompt_render:ctx"
    result = "prompt_render:result"
    return topo_sort_modules([
        _Build(_build_prompt_render, slot="prompt_render.build", produces=(ctx,)),
        _Emit(bus, slot="prompt_render.emit", requires=("prompt_render.build",), data_slot=ctx),
        _RenderMessages(slot="prompt_render.render", requires=("prompt_render.emit",), data_slot=ctx, result_slot=result),
        _Return(slot="prompt_render.return", requires=("prompt_render.render",), data_slot=result),
    ])


def _frame_factory(inp: Any) -> TurnFrame:
    return TurnFrame(input=inp)


# ── TurnLifecycle 门面 ───────────────────────────────────


class TurnLifecycle:
    """一轮对话的生命周期钩子链。

    每个阶段都是独立的 Phase，各自可被 EventBus 上的处理器扩展：
        - 门阶段（before_turn / before_reasoning / prompt_render /
          before_step / after_reasoning）通过 emit 注册，可改写或中止
        - 观察阶段（after_step / after_turn）通过 observe 注册，只读快照
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self.before_turn = Phase(
            _gate_modules("before_turn", bus, _build_before_turn),
            frame_factory=_frame_factory,
        )
        self.before_reasoning = Phase(
            _gate_modules("before_reasoning", bus, _build_before_reasoning),
            frame_factory=_frame_factory,
        )
        self.prompt_render = Phase(
            _prompt_render_modules(bus),
            frame_factory=_frame_factory,
        )
        self.before_step = Phase(
            _gate_modules("before_step", bus, _build_before_step),
            frame_factory=_frame_factory,
        )
        self.after_step = Phase(
            _tap_modules("after_step", bus, _build_after_step),
            frame_factory=_frame_factory,
        )
        self.after_reasoning = Phase(
            _gate_modules("after_reasoning", bus, _build_after_reasoning),
            frame_factory=_frame_factory,
        )
        self.after_turn = Phase(
            _tap_modules("after_turn", bus, _build_after_turn),
            frame_factory=_frame_factory,
        )

    # ── 各阶段执行入口 ─────────────────────────────────

    async def run_before_turn(
        self, msg: InboundMessage, memory_block: str = ""
    ) -> BeforeTurnCtx:
        return await self.before_turn.run(BeforeTurnInput(msg, memory_block))

    async def run_before_reasoning(
        self, bt: BeforeTurnCtx
    ) -> BeforeReasoningCtx:
        return await self.before_reasoning.run(bt)

    async def run_prompt_render(
        self,
        *,
        channel: str,
        chat_id: str,
        content: str,
        system_parts: list[str],
        history: list[dict[str, Any]],
        memory_block: str = "",
    ) -> list[dict[str, Any]]:
        return await self.prompt_render.run(
            PromptRenderInput(channel, chat_id, content, system_parts, history, memory_block)
        )

    async def run_before_step(
        self,
        *,
        channel: str,
        chat_id: str,
        iteration: int,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict] | None,
    ) -> BeforeStepCtx:
        return await self.before_step.run(
            BeforeStepInput(channel, chat_id, iteration, messages, tool_schemas)
        )

    async def run_after_step(
        self,
        *,
        channel: str,
        chat_id: str,
        iteration: int,
        tool_names: tuple[str, ...],
        has_more: bool,
    ) -> AfterStepCtx:
        return await self.after_step.run(
            AfterStepInput(channel, chat_id, iteration, tool_names, has_more)
        )

    async def run_after_reasoning(
        self,
        *,
        channel: str,
        chat_id: str,
        content: str,
        reply: str,
        tools_used: tuple[str, ...],
    ) -> AfterReasoningCtx:
        return await self.after_reasoning.run(
            AfterReasoningInput(channel, chat_id, content, reply, tools_used)
        )

    async def run_after_turn(
        self,
        *,
        channel: str,
        chat_id: str,
        content: str,
        reply: str,
        tools_used: tuple[str, ...],
        will_dispatch: bool = True,
    ) -> AfterTurnCtx:
        return await self.after_turn.run(
            AfterTurnInput(channel, chat_id, content, reply, tools_used, will_dispatch)
        )

    def inspect(self) -> str:
        """打印所有阶段的执行顺序与依赖树（调试用）。"""
        phases = [
            ("before_turn", self.before_turn),
            ("before_reasoning", self.before_reasoning),
            ("prompt_render", self.prompt_render),
            ("before_step", self.before_step),
            ("after_step", self.after_step),
            ("after_reasoning", self.after_reasoning),
            ("after_turn", self.after_turn),
        ]
        lines: list[str] = []
        for name, phase in phases:
            lines.append(f"[{name}]")
            lines.append(_inspect_phase_modules(phase))
            lines.append("")
        return "\n".join(lines)


def _inspect_phase_modules(phase: Phase[Any, Any, TurnFrame]) -> str:
    from agent.lifecycle.phase import inspect_phase

    return inspect_phase(phase._modules)
