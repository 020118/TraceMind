""" Phase 引擎测试

运行方式：
    pytest tests/test_phase.py -v
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from agent.lifecycle.phase import (
    Phase,
    PhaseFrame,
    append_string_exports,
    collect_prefixed_slots,
    topo_sort_modules,
)


@dataclass
class Frame(PhaseFrame[object, object]):
    pass


def _mod(slot: str, requires: tuple[str, ...] = (), produces: tuple[str, ...] = ()):
    """构造一个最小模块，仅声明 slot/requires/produces，run 为 no-op。"""

    class _M:
        pass

    m = _M()
    m.slot = slot
    m.requires = requires
    m.produces = produces

    async def run(frame: Frame) -> Frame:
        return frame

    m.run = run
    return m


# ── 拓扑排序 ────────────────────────────────────────────


def test_topo_sort_respects_dependencies():
    mods = [
        _mod("a.mod2", requires=("a.mod1",)),
        _mod("a.mod1"),
    ]
    sorted_mods = topo_sort_modules(mods)
    assert [m.slot for m in sorted_mods] == ["a.mod1", "a.mod2"]


def test_topo_sort_detects_cycle():
    mods = [
        _mod("a.m1", requires=("a.m2",)),
        _mod("a.m2", requires=("a.m1",)),
    ]
    with pytest.raises(RuntimeError):
        topo_sort_modules(mods)


def test_topo_sort_missing_slot_raises():
    with pytest.raises(RuntimeError):
        topo_sort_modules([_mod("")])


def test_topo_sort_disables_missing_module_dependency():
    """依赖了不存在模块的模块会被禁用，不影响其他模块。"""
    mods = [
        _mod("a.orphan", requires=("a.nonexistent",)),
        _mod("a.ok"),
    ]
    sorted_mods = topo_sort_modules(mods)
    assert [m.slot for m in sorted_mods] == ["a.ok"]


# ── Phase 执行 ──────────────────────────────────────────


def test_phase_runs_modules_in_order_and_returns_output():
    order: list[str] = []

    class Build:
        slot = "p.build"
        requires: tuple[str, ...] = ()
        produces = ("p:ctx",)

        async def run(self, frame):
            order.append("build")
            frame.slots["p:ctx"] = frame.input
            return frame

    class Emit:
        slot = "p.emit"
        requires = ("p.build",)
        produces = ("p:ctx",)

        async def run(self, frame):
            order.append("emit")
            return frame

    class Return:
        slot = "p.return"
        requires = ("p.emit",)
        produces: tuple[str, ...] = ()

        async def run(self, frame):
            order.append("return")
            frame.output = frame.slots["p:ctx"]
            return frame

    phase = Phase([Build(), Emit(), Return()], frame_factory=lambda i: Frame(input=i))
    out = asyncio.run(phase.run("hello"))
    assert out == "hello"
    assert order == ["build", "emit", "return"]


def test_phase_raises_when_no_output():
    class NoOutput:
        slot = "p.noop"
        requires: tuple[str, ...] = ()
        produces: tuple[str, ...] = ()

        async def run(self, frame):
            return frame

    phase = Phase([NoOutput()], frame_factory=lambda i: Frame(input=i))
    with pytest.raises(RuntimeError):
        asyncio.run(phase.run("x"))


# ── 辅助函数 ────────────────────────────────────────────


def test_collect_prefixed_slots():
    slots = {
        "reasoning:extra_hint:foo": "bar",
        "other:key": "x",
        "reasoning:extra_hint:baz": "qux",
    }
    result = collect_prefixed_slots(slots, "reasoning:extra_hint:")
    assert result == {"foo": "bar", "baz": "qux"}


def test_append_string_exports():
    target: list[str] = []
    append_string_exports(target, {"a": "hello", "b": ["x", "y"], "c": 123})
    assert target == ["hello", "x", "y"]
