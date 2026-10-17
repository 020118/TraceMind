""" Agent 层级的事件类型（供 EventBus 使用） """

from dataclasses import dataclass


@dataclass
class LLMResponseEvent:
    """LLM 回复完成事件（用于 observe 日志/指标）"""
    channel: str
    chat_id: str
    query: str
    response: str


@dataclass
class AfterSendEvent:
    """消息已发送事件（用于 fanout 多渠道广播）"""
    channel: str
    chat_id: str
    content: str
