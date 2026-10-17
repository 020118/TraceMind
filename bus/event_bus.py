"""event_bus 垫片 —— 复用 myAgent 的 EventBus 实现，避免出现两套事件总线类"""

from agent.bus.event_bus import EventBus

__all__ = ["EventBus"]
