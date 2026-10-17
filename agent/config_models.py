""" 配置数据模型 —— 完整对接 config.toml """

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMProfile:
    """单个 LLM 模型配置"""
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    enable_thinking: bool = False
    multimodal: bool = False


@dataclass
class LLMConfig:
    """LLM 整体配置"""
    provider: str = "openai"
    main: LLMProfile = field(default_factory=LLMProfile)
    fast: LLMProfile | None = None
    vl: LLMProfile | None = None


@dataclass
class MemoryEmbeddingConfig:
    """向量模型配置"""
    model: str = ""
    api_key: str = ""
    base_url: str = ""


@dataclass
class MemoryRetrievalConfig:
    """V2 记忆混合检索配置"""
    top_k: int = 6
    score_threshold: float = 0.35
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    enabled: bool = False
    engine: str = ""
    embedding: MemoryEmbeddingConfig = field(default_factory=MemoryEmbeddingConfig)
    retrieval: MemoryRetrievalConfig = field(default_factory=MemoryRetrievalConfig)


@dataclass
class RAGConfig:
    """文档知识库配置（与个人记忆相互独立）。"""
    enabled: bool = False
    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k: int = 6
    rrf_k: int = 60
    embedding: MemoryEmbeddingConfig = field(default_factory=MemoryEmbeddingConfig)
    vector_store: "RAGVectorStoreConfig" = field(default_factory=lambda: RAGVectorStoreConfig())


@dataclass
class RAGVectorStoreConfig:
    """外部向量数据库与影子评测配置。"""
    provider: str = "sqlite"
    url: str = ""
    api_key: str = ""
    collection: str = "agent_knowledge_v1"
    timeout_seconds: float = 10.0
    read_from_qdrant: bool = True
    shadow_evaluation: bool = True
    sync_on_start: bool = True


@dataclass
class TelegramConfig:
    """Telegram 通道配置"""
    token: str = ""
    allow_from: list[str] = field(default_factory=list)


@dataclass
class ChannelsConfig:
    """通道配置"""
    telegram: TelegramConfig | None = None


@dataclass
class RuntimeConfig:
    """运行时配置"""
    workspace: str = "~/.akashic/workspace"


@dataclass
class AgentConfig:
    """根配置 —— 对应 config.toml 完整结构"""

    # LLM
    llm: LLMConfig = field(default_factory=LLMConfig)

    # 记忆
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # 文档知识库（RAG）
    rag: RAGConfig = field(default_factory=RAGConfig)

    # 通道
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)

    # 运行时
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # 主动推送（proactive_v2），原始 TOML 字典，交给 load_proactive_config 解析
    proactive: dict = field(default_factory=dict)

    # 通用
    system_prompt: str = "You are a helpful assistant."
    max_tokens: int = 8192
    max_iterations: int = 10
    tool_timeout_seconds: float = 30.0
    memory_window: int = 40

    @classmethod
    def load(cls, path: str | Path = "config.toml") -> AgentConfig:
        """从 TOML 文件加载配置"""
        from agent.config import load_config
        return load_config(path)
