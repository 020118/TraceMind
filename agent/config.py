""" 配置加载 —— 从 config.toml 解析为 AgentConfig """

from __future__ import annotations

import re
import os
import tomllib
from pathlib import Path

from agent.config_models import (
    AgentConfig,
    LLMConfig,
    LLMProfile,
    MemoryConfig,
    MemoryEmbeddingConfig,
    MemoryRetrievalConfig,
    RAGConfig,
    RAGVectorStoreConfig,
    ChannelsConfig,
    TelegramConfig,
    RuntimeConfig,
)


PROVIDER_PRESETS: dict[str, str] = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
}


def load_config(path: str | Path = "config.toml") -> AgentConfig:
    """解析 TOML 配置文件，返回 AgentConfig"""
    data = _load_toml(path)

    # ── LLM ────────────────────────────────────────────
    llm_raw = _as_dict(data.get("llm"))
    provider = str(llm_raw.get("provider", "openai"))

    llm = LLMConfig(
        provider=provider,
        main=_parse_llm_profile(llm_raw, "main", provider),
        fast=_parse_llm_profile(llm_raw, "fast", provider),
        vl=_parse_llm_profile(llm_raw, "vl", provider),
    )

    # ── Memory ─────────────────────────────────────────
    mem_raw = _as_dict(data.get("memory"))
    embed_raw = _as_dict(mem_raw.get("embedding"))
    retrieval_raw = _as_dict(mem_raw.get("retrieval"))
    memory = MemoryConfig(
        enabled=bool(mem_raw.get("enabled", False)),
        engine=str(mem_raw.get("engine", "")),
        embedding=MemoryEmbeddingConfig(
            model=str(embed_raw.get("model", "")),
            api_key=_resolve(str(embed_raw.get("api_key", ""))),
            base_url=str(embed_raw.get("base_url", "")),
        ),
        retrieval=MemoryRetrievalConfig(
            top_k=_clamp_int(retrieval_raw.get("top_k", 6), 1, 20),
            score_threshold=_clamp_float(
                retrieval_raw.get("score_threshold", 0.35), 0.0, 1.0
            ),
            semantic_weight=max(
                0.0, float(retrieval_raw.get("semantic_weight", 0.7))
            ),
            keyword_weight=max(
                0.0, float(retrieval_raw.get("keyword_weight", 0.3))
            ),
        ),
    )

    # ── RAG 文档知识库 ────────────────────────────────
    rag_raw = _as_dict(data.get("rag"))
    rag_embed_raw = _as_dict(rag_raw.get("embedding"))
    vector_store_raw = _as_dict(rag_raw.get("vector_store"))
    # 默认复用 memory 的 Embedding 配置，避免同一个向量服务配置两遍。
    rag = RAGConfig(
        enabled=bool(rag_raw.get("enabled", False)),
        chunk_size=_clamp_int(rag_raw.get("chunk_size", 1200), 200, 8000),
        chunk_overlap=_clamp_int(rag_raw.get("chunk_overlap", 150), 0, 2000),
        top_k=_clamp_int(rag_raw.get("top_k", 6), 1, 20),
        rrf_k=_clamp_int(rag_raw.get("rrf_k", 60), 1, 1000),
        embedding=MemoryEmbeddingConfig(
            model=str(rag_embed_raw.get("model", memory.embedding.model)),
            api_key=_resolve(
                str(rag_embed_raw.get("api_key", memory.embedding.api_key))
            ),
            base_url=str(rag_embed_raw.get("base_url", memory.embedding.base_url)),
        ),
        vector_store=RAGVectorStoreConfig(
            provider=str(vector_store_raw.get("provider", "sqlite")).strip().lower(),
            url=_resolve(str(vector_store_raw.get("url", ""))),
            api_key=_resolve(str(vector_store_raw.get("api_key", ""))),
            collection=str(
                vector_store_raw.get("collection", "agent_knowledge_v1")
            ).strip(),
            timeout_seconds=_clamp_float(
                vector_store_raw.get("timeout_seconds", 10.0), 1.0, 60.0
            ),
            read_from_qdrant=bool(
                vector_store_raw.get("read_from_qdrant", True)
            ),
            shadow_evaluation=bool(
                vector_store_raw.get("shadow_evaluation", True)
            ),
            sync_on_start=bool(vector_store_raw.get("sync_on_start", True)),
        ),
    )

    # ── Channels ───────────────────────────────────────
    ch_raw = _as_dict(data.get("channels"))
    tg_raw = _as_dict(ch_raw.get("telegram"))
    channels = ChannelsConfig(
        telegram=TelegramConfig(
            token=_resolve(str(tg_raw.get("token", ""))),
            allow_from=list(tg_raw.get("allow_from", [])),
        ) if tg_raw else None,
    )

    # ── Runtime ────────────────────────────────────────
    rt_raw = _as_dict(data.get("runtime"))
    runtime = RuntimeConfig(
        workspace=str(rt_raw.get("workspace", "~/.akashic/workspace")),
    )

    # ── Proactive（原始字典，由 proactive_v2.config_loader 解析） ──
    proactive_raw = _as_dict(data.get("proactive"))

    # ── Agent 通用配置 ──────────────────────────────────
    agent_raw = _as_dict(data.get("agent"))
    agent_ctx = _as_dict(agent_raw.get("context"))

    return AgentConfig(
        llm=llm,
        memory=memory,
        rag=rag,
        channels=channels,
        runtime=runtime,
        proactive=proactive_raw,
        system_prompt=str(
            agent_raw.get("system_prompt", data.get("system_prompt", "You are a helpful assistant."))
        ),
        max_tokens=int(agent_raw.get("max_tokens", data.get("max_tokens", 8192))),
        max_iterations=int(agent_raw.get("max_iterations", data.get("max_iterations", 10))),
        tool_timeout_seconds=_clamp_float(
            agent_raw.get("tool_timeout_seconds", data.get("tool_timeout_seconds", 30.0)),
            1.0,
            300.0,
        ),
        memory_window=int(
            agent_ctx.get("memory_window", data.get("memory_window", 40))
        ),
    )


# ── 内部辅助 ────────────────────────────────────────


def _parse_llm_profile(data: dict, key: str, provider: str) -> LLMProfile | None:
    """解析 [llm.main] / [llm.fast] / [llm.vl] 节"""
    section = _as_dict(data.get(key))
    if not section or not section.get("model"):
        return None

    api_key = _resolve(str(section.get("api_key", data.get("api_key", ""))))
    base_url = str(
        section.get("base_url")
        or data.get("base_url")
        or PROVIDER_PRESETS.get(provider, "")
    )

    return LLMProfile(
        model=str(section["model"]),
        api_key=api_key,
        base_url=base_url,
        enable_thinking=bool(section.get("enable_thinking", False)),
        multimodal=bool(section.get("multimodal", False)),
    )


def _load_toml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    if path.suffix.lower() != ".toml":
        raise ValueError(f"配置仅支持 TOML 格式: {path.suffix}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _clamp_int(value: object, minimum: int, maximum: int) -> int:
    return min(max(int(value), minimum), maximum)


def _clamp_float(value: object, minimum: float, maximum: float) -> float:
    return min(max(float(value), minimum), maximum)


def _resolve(value: str) -> str:
    """展开 ${VAR_NAME} 环境变量占位符"""
    def _replace(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(0))
    resolved = re.sub(r"\$\{(\w+)\}", _replace, value)

    # 若仍是未展开的占位符，尝试从 ~/.akashic/workspace/memory/<VAR> 读取
    m = re.fullmatch(r"\$\{(\w+)\}", resolved)
    if m:
        key_file = Path.home() / ".akashic" / "workspace" / "memory" / m.group(1)
        if key_file.exists():
            resolved = key_file.read_text(encoding="utf-8").strip()
        else:
            # 未配置的占位符不能被误当成有效 API key；让启动阶段给出明确错误。
            resolved = ""
    return resolved
