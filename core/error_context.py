"""error_context 垫片 —— 复用 myAgent 已有的 ContextVar，保证全项目只有一个实例"""

from agent.core.error_context import current_session_key

__all__ = ["current_session_key"]
