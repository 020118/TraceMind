"""events 垫片 —— 复用 myAgent 的消息事件模型（InboundMessage/OutboundMessage）"""

from agent.bus.events import InboundMessage, OutboundMessage

__all__ = ["InboundMessage", "OutboundMessage"]
