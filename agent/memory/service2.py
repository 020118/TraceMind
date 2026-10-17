""" 记忆服务层 V2 —— 适配 MemoryStoreV2（六层记忆 + 两阶段提交）

职责：
    - 适配 MemoryStoreV2 的六层记忆架构
    - 事实萃取 → 追加到 PENDING.md（幂等）
    - 记忆上下文构建（读取 MEMORY.md）
    - 暴露 Consolidation 触发接口
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from agent.llm.client import LLMClient, LLMResponse
from agent.memory.retrieval2 import HybridMemoryRetriever
from agent.memory.store2 import MemoryStoreV2, MemoryStoreConfig

logger = logging.getLogger(__name__)

# 事实萃取 system prompt（保持原有严格约束）
FACT_EXTRACTION_PROMPT = """你是一个记忆萃取助手。从用户的**本轮消息**中提取出关于用户的、值得记住的事实。

核心原则：只基于用户本轮消息原文，禁止使用任何历史信息或AI回复来反推。

硬规则（违反任一就判定为不合格萃取，返回空数组）：
1. 只基于用户**本轮消息原文**提取。AI的回复、历史对话中的信息、你"知道"的用户信息，一律不得作为提取依据
2. 所有具体细节（地名、人名、数字）必须原样保留，禁止修改
3. 必须提取所有用户说出的个人信息，不能遗漏
4. 如果用户本轮消息只是在提问或寻求建议，没有主动透露个人信息，返回空数组 []
5. 每一条事实必须能在用户本轮消息原文中找到逐字依据
6. 去除空字符串和重复项
7. 返回 JSON 数组：["事实1", "事实2", ...]

示例（正例 — 用户本轮说了个人信息）：
用户：我叫张三，住在北京，喜欢吃辣
→ ["用户名叫张三", "用户住在北京", "用户喜欢吃辣"]

用户：我是东莞人，喜欢跑步和打羽毛球
→ ["用户是东莞人", "用户喜欢跑步", "用户喜欢打羽毛球"]

示例（反例 — 用户本轮没透露信息，不应萃取）：
用户：有什么可以去拍照的好地方吗？
→ []  ← 用户只是在提问，没有说出个人信息

用户：今天天气真好
→ []  ← 一次性信息，没有个人信息

用户：推荐一下好吃的
→ []  ← 没有个人信息

用户：推荐个咖啡店
→ []  ← 没有透露个人信息

用户：有什么好的跑步路线
→ []  ← 没有透露个人信息

注意：即使你知道用户住在东莞、喜欢跑步，但如果用户本轮消息没提这些，就不能提取。
"""

MEMORY_CONTEXT_TEMPLATE = """
以下是与当前话题相关的参考信息：

{memories}
""".strip()


class MemoryServiceV2:
    """记忆服务 V2 —— 适配六层记忆架构"""

    def __init__(
        self,
        store: MemoryStoreV2,
        fast_llm: LLMClient | None = None,
        retriever: HybridMemoryRetriever | None = None,
    ):
        """
        参数:
            store:         已初始化的 MemoryStoreV2 实例
            fast_llm:      用于事实萃取的轻量 LLM（推荐用 qwen-flash）
                           None 表示不启用自动萃取
        """
        self._store = store
        self._fast_llm = fast_llm
        self._retriever = retriever

    @property
    def store(self) -> MemoryStoreV2:
        return self._store

    # ── 事实萃取 ───────────────────────────────────────

    async def extract_facts(
        self,
        user_message: str,
        assistant_message: str = "",  # 保留参数兼容性，不传给 LLM
    ) -> list[str]:
        """从用户本轮消息中萃取事实

        只基于用户原文，不使用 AI 回复或历史信息反推。

        参数:
            user_message:      用户本轮消息（唯一依据）
            assistant_message: AI 回复（仅用于日志，不作为提取依据）
        """
        if not self._fast_llm:
            return []

        user_text = (user_message or "").strip()
        if not user_text:
            return []

        try:
            resp: LLMResponse = await self._fast_llm.chat(
                messages=[
                    {"role": "system", "content": FACT_EXTRACTION_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                tools=None,
            )
            raw = (resp.content or "").strip()
            # 清理可能的 markdown 代码块标记
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            facts = json.loads(raw) if raw else []
            if isinstance(facts, list) and facts:
                logger.info("萃取到 %d 条事实: %s", len(facts), facts[:3])
            return facts if isinstance(facts, list) else []
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("事实萃取失败: %s", e)
            return []

    # ── 记忆写入 ───────────────────────────────────────

    async def add_fact(self, fact: str, source_ref: str) -> bool:
        """添加一条萃取事实到 PENDING.md（幂等）

        参数:
            fact:       萃取出的单条事实文本
            source_ref: 来源标识（建议用 session_key 或 "session_key_turn_N"）
        """
        if not fact or not fact.strip():
            return False
        return self._store.append_pending(fact.strip(), source_ref)

    # ── 记忆上下文构建 ─────────────────────────────────

    def build_memory_context(self) -> str:
        """兼容旧调用：构建完整记忆上下文块。"""
        return self._store.get_memory_context()

    async def retrieve_memory_context(self, query: str) -> str:
        """只返回与当前问题相关的 V2 记忆。"""
        if self._retriever is None:
            return self.build_memory_context()
        return await self._retriever.build_context(query)

    # ── Consolidation 触发 ────────────────────────────

    async def consolidate(self, llm_client) -> dict:
        """执行 Consolidation 归档：PENDING → MEMORY.md

        参数:
            llm_client: 用于归纳总结的 LLM 客户端（建议用主模型）

        返回:
            {"success": bool, "facts_count": int, "error": str|None}
        """
        # Phase-1: 原子快照
        snapshot = await self._store.snapshot_pending()
        if not snapshot:
            return {"success": True, "facts_count": 0, "error": "无待归档内容"}

        # LLM 归纳总结
        long_term = self._store.read_long_term()
        prompt = f"""请将以下增量事实合并到长期记忆中，去重、合并同类项，保持简洁：

## 现有长期记忆
{long_term if long_term else "(空)"}

## 新增增量事实
{snapshot}

请输出合并后的完整长期记忆（Markdown 格式），不要包含额外解释。"""

        try:
            resp = await llm_client.chat(
                messages=[
                    {"role": "system", "content": "你是记忆整理助手，擅长归纳总结用户画像。"},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
            )
            merged = (resp.content or "").strip()
            if not merged:
                raise ValueError("LLM 返回空内容")

            # 写入新 MEMORY.md
            self._store.write_long_term(merged)

            # 记录 consolidation 到 HISTORY.md
            import time
            source_ref = f"consolidation_{int(time.time())}"
            self._store.append_history_once(
                text=f"Consolidated at {datetime.now().isoformat()}",
                source_ref=source_ref,
                kind="consolidation",
            )

            # Phase-2 成功提交
            self._store.commit_pending_snapshot()

            # 统计事实条数
            facts_count = len([l for l in snapshot.split("\n") if l.strip()])

            return {"success": True, "facts_count": facts_count, "error": None}

        except Exception as e:
            logger.exception("Consolidation 失败")
            # Phase-2 失败：回滚
            self._store.rollback_pending_snapshot()
            return {"success": False, "facts_count": 0, "error": str(e)}

    # ── 会话历史隔离 ──────────────────────────────────

    def get_session_history(self, session_key: str) -> list[dict]:
        return self._store.get_session_history(session_key)

    def append_session_history(self, session_key: str, role: str, content: str) -> None:
        self._store.append_session_history(session_key, role, content)

    def get_session_history_for_prompt(self, session_key: str) -> list[dict]:
        return self._store.get_session_history_for_prompt(session_key)

    async def close(self) -> None:
        """关闭检索客户端与底层存储。"""
        if self._retriever is not None:
            await self._retriever.close()
        await self._store.close()


def create_memory_service_v2(
    workspace: Path,
    fast_llm_config: dict | None = None,
    embedding_config: dict | None = None,
    retrieval_config: dict | None = None,
    max_history_chars: int = 100000,
) -> MemoryServiceV2:
    """创建 MemoryServiceV2 实例

    参数:
        workspace: 工作区路径
        fast_llm_config: 轻量 LLM 配置
            {"api_key": "...", "base_url": "...", "model": "qwen-flash"}
        embedding_config: 向量模型配置；未配置时自动使用关键词检索
        retrieval_config: top_k、阈值与混合权重
        max_history_chars: 单会话历史最大字符数
    """
    from agent.llm.client import LLMClient

    config = MemoryStoreConfig(
        workspace=Path(workspace),
        max_history_chars=max_history_chars,
    )
    store = MemoryStoreV2(config)

    fast_llm = None
    if fast_llm_config:
        fast_llm = LLMClient(
            api_key=fast_llm_config["api_key"],
            base_url=fast_llm_config["base_url"],
            model=fast_llm_config["model"],
        )

    embedding_config = embedding_config or {}
    retrieval_config = retrieval_config or {}
    retriever = HybridMemoryRetriever(
        store,
        workspace=Path(workspace),
        api_key=str(embedding_config.get("api_key", "")),
        base_url=str(embedding_config.get("base_url", "")),
        model=str(embedding_config.get("model", "")),
        top_k=int(retrieval_config.get("top_k", 6)),
        score_threshold=float(retrieval_config.get("score_threshold", 0.35)),
        semantic_weight=float(retrieval_config.get("semantic_weight", 0.7)),
        keyword_weight=float(retrieval_config.get("keyword_weight", 0.3)),
    )

    return MemoryServiceV2(store=store, fast_llm=fast_llm, retriever=retriever)
