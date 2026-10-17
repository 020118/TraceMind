""" Phase 引擎 —— 声明式、可编排的钩子链

把一个处理阶段（Phase）拆成若干模块（PhaseModule），每个模块声明：

    slot:      str              — 模块在管线中的唯一标识（形如 "before_turn.xxx"）
    requires:  tuple[str, ...]  — 依赖的 slot（拓扑排序据此编排顺序）
    produces:  tuple[str, ...]  — 产出到 frame.slots 的数据槽位

约定：
    - 模块 slot 用 "." 分隔（内置阶段前缀，如 "before_turn."）
    - 数据 slot 用 ":" 分隔（如 "reasoning:ctx"），仅用于跨模块传递中间结果
    - Phase 启动时按 requires 做拓扑排序，并校验依赖是否闭合

使用方式：
    phase = Phase(modules=[...], frame_factory=lambda i: TurnFrame(input=i))
    output = await phase.run(input)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, cast

logger = logging.getLogger(__name__)

I = TypeVar("I")
O = TypeVar("O")
F = TypeVar("F", bound="PhaseFrame[Any, Any]")


@dataclass
class PhaseFrame(Generic[I, O]):
    """一次 Phase 执行的载体：input 是输入，output 是最终产出，slots 存中间数据。"""

    input: I
    slots: dict[str, Any] = field(default_factory=dict)
    output: O | None = None


class PhaseModule(Protocol[F]):
    """模块约定：声明 slot / requires / produces，实现 run(frame) -> frame。"""

    async def run(self, frame: F) -> F:
        ...


class SlotModule(Protocol):
    slot: str


def _empty_slots() -> dict[str, Any]:
    return {}


def collect_prefixed_slots(
    slots: Mapping[str, object],
    prefix: str,
    *,
    reserved: Collection[str] = (),
) -> dict[str, object]:
    """收集 slots 中以 prefix 开头的键，去掉前缀后作为名字返回。"""
    values: dict[str, object] = {}
    reserved_fields = set(reserved)
    for key, value in slots.items():
        if not key.startswith(prefix):
            continue
        field_name = key.removeprefix(prefix)
        if not field_name or field_name in reserved_fields:
            continue
        values[field_name] = value
    return values


def append_string_exports(target: list[str], exports: Mapping[str, object]) -> None:
    """把 export slot 中的字符串（或字符串列表）追加到 target。"""
    for key, value in exports.items():
        if isinstance(value, str) and value.strip():
            target.append(value)
        elif isinstance(value, list):
            for item in cast(list[object], value):
                if isinstance(item, str) and item.strip():
                    target.append(item)
                elif item is not None:
                    logger.warning(
                        "忽略非字符串 slot export: key=%s type=%s",
                        key,
                        type(item).__name__,
                    )
        elif value is not None:
            logger.warning(
                "忽略非字符串 slot export: key=%s type=%s",
                key,
                type(value).__name__,
            )


# 内置阶段前缀：用于排序稳定性与依赖校验
_BUILTIN_SLOT_PREFIXES = (
    "before_turn.",
    "before_reasoning.",
    "prompt_render.",
    "before_step.",
    "after_step.",
    "after_reasoning.",
    "after_turn.",
)


def _is_builtin_slot(slot: str) -> bool:
    return slot.startswith(_BUILTIN_SLOT_PREFIXES)


def _is_module_slot(slot: str) -> bool:
    return "." in slot and ":" not in slot


def _module_requires(
    module: object,
    known_slots: Mapping[str, object],
) -> tuple[str, ...]:
    requires = tuple(str(slot) for slot in getattr(module, "requires", ()))
    return tuple(slot for slot in requires if slot in known_slots)


def _missing_module_requires(
    module: object,
    active_slots: set[str],
) -> tuple[str, ...]:
    return tuple(
        req
        for req in (str(slot) for slot in getattr(module, "requires", ()))
        if _is_module_slot(req) and req not in active_slots
    )


def _active_module_slots(slot_map: Mapping[str, object]) -> set[str]:
    """过滤掉依赖缺失（引用不存在模块）的模块 slot。"""
    active = set(slot_map)
    while True:
        disabled: set[str] = set()
        for slot, module in slot_map.items():
            if slot not in active or _is_builtin_slot(slot):
                continue
            missing = _missing_module_requires(module, active)
            if missing:
                logger.warning(
                    "Phase 模块依赖不存在，已禁用模块: module=%s requires=%s",
                    slot,
                    ", ".join(missing),
                )
                disabled.add(slot)
        if not disabled:
            return active
        active -= disabled


def topo_sort_modules(modules: Sequence[object]) -> list[object]:
    """按 requires 依赖关系做拓扑排序，返回有序模块列表。"""
    slot_map: dict[str, object] = {}
    slot_order: dict[str, int] = {}
    for index, module in enumerate(modules):
        slot = getattr(module, "slot", None)
        if not isinstance(slot, str) or not slot:
            raise RuntimeError(f"模块缺少 slot 声明: {type(module).__name__}")
        if slot in slot_map:
            raise RuntimeError(f"模块 slot 重复: {slot}")
        slot_map[slot] = module
        slot_order[slot] = index
    active_slots = _active_module_slots(slot_map)
    slot_map = {slot: module for slot, module in slot_map.items() if slot in active_slots}

    in_degree = {slot: 0 for slot in slot_map}
    dependents: dict[str, list[str]] = {slot: [] for slot in slot_map}
    for slot, module in slot_map.items():
        for req in _module_requires(module, slot_map):
            in_degree[slot] += 1
            dependents[req].append(slot)

    queue = [slot for slot, degree in in_degree.items() if degree == 0]
    sorted_modules: list[object] = []
    while queue:
        queue.sort(key=lambda item: (_is_builtin_slot(item), slot_order[item]))
        slot = queue.pop(0)
        sorted_modules.append(slot_map[slot])
        for dependent in dependents[slot]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_modules) != len(slot_map):
        unresolved = sorted(slot for slot, degree in in_degree.items() if degree > 0)
        raise RuntimeError(f"模块循环依赖: {', '.join(unresolved)}")
    return sorted_modules


def _disable_modules_with_missing_module_dependencies(
    modules: Sequence[object],
) -> list[object]:
    """过滤掉依赖了不存在模块的模块（用于 Phase 初始化）。"""
    module_slots = {
        str(slot)
        for slot in (getattr(module, "slot", None) for module in modules)
        if isinstance(slot, str) and slot
    }
    active = set(module_slots)
    disabled: set[str] = set()
    while True:
        current: set[str] = set()
        for module in modules:
            slot = getattr(module, "slot", None)
            if not isinstance(slot, str) or not slot:
                continue
            if slot not in active or _is_builtin_slot(slot):
                continue
            missing = _missing_module_requires(module, active)
            if missing:
                logger.warning(
                    "Phase 模块依赖不存在，已禁用模块: module=%s requires=%s",
                    slot,
                    ", ".join(missing),
                )
                current.add(slot)
        if not current:
            break
        disabled |= current
        active -= current
    return [
        module
        for module in modules
        if getattr(module, "slot", None) not in disabled
    ]


def render_dependency_tree(modules: Sequence[object]) -> str:
    sorted_modules = cast(list[SlotModule], topo_sort_modules(modules))
    slot_map: dict[str, SlotModule] = {module.slot: module for module in sorted_modules}
    children: dict[str, list[str]] = {slot: [] for slot in slot_map}
    in_degree = {slot: 0 for slot in slot_map}

    for slot, module in slot_map.items():
        for req in _module_requires(module, slot_map):
            children[req].append(slot)
            in_degree[slot] += 1

    roots = [slot for slot, degree in in_degree.items() if degree == 0]
    lines: list[str] = []
    visited: set[str] = set()
    for index, slot in enumerate(roots):
        _render_tree_node(
            lines,
            slot,
            children,
            slot_map,
            visited,
            prefix="",
            is_last=index == len(roots) - 1,
        )
    return "\n".join(lines)


def inspect_phase(modules: Sequence[object]) -> str:
    sorted_modules = cast(list[SlotModule], topo_sort_modules(modules))
    chain = "\n".join(
        f"  {index:2d}. {module.slot}"
        for index, module in enumerate(sorted_modules)
    )
    tree = render_dependency_tree(sorted_modules)
    return f"执行顺序:\n{chain}\n\n依赖树:\n{tree}"


def _module_label(module: SlotModule) -> str:
    slot = module.slot
    tag = "[B]" if _is_builtin_slot(slot) else "[P]"
    return f"{tag} {slot}"


def _render_tree_node(
    lines: list[str],
    slot: str,
    children: Mapping[str, list[str]],
    slot_map: Mapping[str, SlotModule],
    visited: set[str],
    *,
    prefix: str,
    is_last: bool,
) -> None:
    if slot in visited:
        return
    visited.add(slot)
    connector = "└── " if is_last else "├── "
    module = slot_map[slot]
    lines.append(f"{prefix}{connector}{_module_label(module)}")
    _append_tree_details(lines, module, prefix=prefix, is_last=is_last)
    child_prefix = prefix + ("    " if is_last else "│   ")
    child_slots = children.get(slot, [])
    for index, child in enumerate(child_slots):
        _render_tree_node(
            lines,
            child,
            children,
            slot_map,
            visited,
            prefix=child_prefix,
            is_last=index == len(child_slots) - 1,
        )


def _append_tree_details(
    lines: list[str],
    module: SlotModule,
    *,
    prefix: str,
    is_last: bool,
) -> None:
    indent = prefix + ("    " if is_last else "│   ")
    requires = tuple(str(slot) for slot in getattr(module, "requires", ()))
    produces = tuple(str(slot) for slot in getattr(module, "produces", ()))
    if requires:
        lines.append(f"{indent}← requires: {', '.join(requires)}")
    if produces:
        lines.append(f"{indent}→ produces: {', '.join(produces)}")


class Phase(Generic[I, O, F]):
    """一个处理阶段：按依赖顺序串行执行模块链，最终产出 output。

    参数:
        modules:        参与该阶段的模块（含内置与插件/用户注册模块）
        frame_factory:  从 input 构造 frame 的工厂
    """

    def __init__(
        self,
        modules: Sequence[PhaseModule[F]],
        *,
        frame_factory: Callable[[I], F],
    ) -> None:
        self._modules = _disable_modules_with_missing_module_dependencies(modules)
        self._frame_factory = frame_factory
        self._validate()

    async def run(self, input: I) -> O:
        frame = self._frame_factory(input)
        for module in self._modules:
            frame = await module.run(frame)
        if frame.output is None:
            raise RuntimeError("Phase 模块链未产生 output")
        return frame.output

    def _validate(self) -> None:
        module_slots = {
            str(slot)
            for slot in (getattr(module, "slot", None) for module in self._modules)
            if isinstance(slot, str) and slot
        }
        provided: set[str] = set()
        for index, module in enumerate(self._modules):
            requires = tuple(getattr(module, "requires", ()))
            produces = tuple(getattr(module, "produces", ()))
            for raw_slot in requires:
                slot = str(raw_slot)
                if slot in module_slots:
                    if slot != getattr(module, "slot", None) and slot not in provided:
                        logger.warning(
                            "Phase 模块依赖未满足: module=%d name=%s requires=%s",
                            index,
                            module.__class__.__name__,
                            slot,
                        )
                    continue
                if _is_module_slot(slot):
                    logger.warning(
                        "Phase 模块依赖不存在: module=%d name=%s requires=%s",
                        index,
                        module.__class__.__name__,
                        slot,
                    )
                    continue
                if slot not in provided:
                    logger.warning(
                        "Phase slot 未闭合: module=%d name=%s requires=%s",
                        index,
                        module.__class__.__name__,
                        slot,
                    )
            module_slot = getattr(module, "slot", None)
            if isinstance(module_slot, str) and module_slot:
                provided.add(module_slot)
            provided.update(str(slot) for slot in produces)
