""" 记忆系统 —— 轻量级语义记忆

使用方式：
    from agent.memory.store import MemoryStore, MemoryItem
    from agent.memory.service import MemoryService, create_memory_service

架构：
    MemoryService（高层：embedding + 搜索 + 萃取）
        │
        ▼
    MemoryStore（底层：SQLite + 余弦相似度）
"""

from agent.memory.store import MemoryStore, MemoryItem
from agent.memory.service import MemoryService, create_memory_service

DEFAULT_SELF_MD = """# 我的自我认知
## 人格与形象
- 我是一个直接、务实、主动参与思考的长期协作伙伴。
- 我优先给出结论，再补充必要细节；不把自己伪装成没有立场的工具。

## 我对当前用户的理解
- 我会从长期记忆中逐步形成对当前用户的理解，不在缺少证据时编造画像。

## 我们关系的定位
- 我与用户的关系以透明、尊重边界和持续协作为基础。"""

__all__ = [
    "MemoryStore",
    "MemoryItem",
    "MemoryService",
    "create_memory_service",
    "DEFAULT_SELF_MD",
]
