""" 生命周期包 —— 应用生命周期管理 + 每轮对话的 Phase 钩子链 """

from agent.lifecycle.manager import LifecycleManager
from agent.lifecycle.phase import Phase, PhaseFrame, topo_sort_modules, inspect_phase
from agent.lifecycle.turn import TurnLifecycle

__all__ = [
    "LifecycleManager",
    "Phase",
    "PhaseFrame",
    "topo_sort_modules",
    "inspect_phase",
    "TurnLifecycle",
]
