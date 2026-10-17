""" 消息总线的事件类型 """

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from enum import Enum

if TYPE_CHECKING:
    from agent.bus.outbound_events import OutboundEvent


class LifecyclePhase(str, Enum):
    """生命周期阶段"""
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"


# 出站消息中用于结构化 UI 负载的 metadata key
OUTBOUND_META_AGENT_UI = "_agent_ui"

# 入站消息中用于运行时控制的 metadata key（in-process 通道使用）
INBOUND_META_RUNTIME_CONTROL = "_runtime_control"
RUNTIME_CONTROL_ACK = "_ack"
RUNTIME_CONTROL_MCP_RELOAD = "mcp_reload"


@dataclass
class InboundMessage:
    """接收自聊天通道的消息

    channel 表示来源（telegram, qq, discord…），
    sender_id 是用户标识，chat_id 是所在聊天/频道标识，
    content 是消息文本，media 是附件 URL 列表。
    metadata 承载通道特定数据（如 message_id、thread_id），
    也可用于 in-process 通道传递运行时控制指令。
    """

    channel: str                # 通道来源: telegram, qq, discord...
    sender_id: str              # 发送者标识
    chat_id: str                # 聊天/频道标识（群聊或私聊）
    content: str                # 消息文本
    timestamp: datetime = field(default_factory=datetime.now)   # 消息时间戳
    media: list[str] = field(default_factory=list)              # 媒体附件 URL
    metadata: dict[str, Any] = field(default_factory=dict)      # 通道特定元数据
    session_key_override: str | None = None                     # 可选会话 key 覆盖

    @property
    def session_key(self) -> str:
        """会话唯一标识 key

        默认格式为 "{channel}:{chat_id}"，可通过 session_key_override 覆盖。
        """
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """发送至聊天通道的消息

    event 字段承载运行时/UI 语义（ProgressEvent、StreamDeltaEvent 等）。
    metadata 为通道路由上下文保留（message_id、thread ids 等），
    也可包含 OUTBOUND_META_AGENT_UI 供富客户端渲染。

    buttons 用于快速回复按钮（嵌套列表：每行为一行按钮，每项为按钮文本）。
    """

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)  #field的作用是每当有新建OutboundMessage时能够创建一个空容器给不需要的对象。
    metadata: dict[str, Any] = field(default_factory=dict)
    buttons: list[list[str]] = field(default_factory=list)
    event: "OutboundEvent | None" = None


@dataclass
class LifecycleEvent:
    """应用生命周期事件（用于 EventBus 的 observe/fanout）"""
    phase: LifecyclePhase
    message: str = ""
