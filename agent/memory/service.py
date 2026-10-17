""" 记忆服务层 —— embedding + 搜索 + 事实萃取

把底层存储（MemoryStore）和上层 LLM 能力粘合在一起。

职责：
    - 调用 embedding API 把文本转成向量
    - 提供"搜索记忆"的高层接口（embed 问题 → 搜 → 返回）
    - 用 LLM 从对话中提取事实（事实萃取）
    - 把相关记忆注入到 LLM 上下文

使用方式：
    service = MemoryService(
        store=MemoryStore("memories.db"),
        embed_api_key="sk-xxx",
        embed_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embed_model="text-embedding-v3",
        fast_llm=fast_llm_client,     # 用于事实萃取的轻量 LLM
    )
    await service.initialize()

    # 搜索
    results = await service.search("用户喜欢吃什么？", user_id="user_1")

    # 萃取事实
    facts = await service.extract_facts("用户：我喜欢吃辣", "AI：推荐了火锅")
    for fact in facts:
        await service.add_memory(fact, source="chat_001")

    # 构建上下文
    memory_block = await service.build_memory_context("用户问题", user_id="user_1")
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from agent.llm.client import LLMClient, LLMResponse
from agent.memory.store import MemoryItem, MemoryStore

logger = logging.getLogger(__name__)

# 事实萃取用的 system prompt
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

# 记忆上下文注入的模板
# 注意：只注入与当前问题高度相关的信息，不相关的内容不注入
MEMORY_CONTEXT_TEMPLATE = """
以下是与当前话题相关的参考信息：

{memories}
"""


class MemoryService:
    """记忆服务 —— 高层操作接口

    组合了 MemoryStore（存储）和 embedding/LLM API（智能处理）。
    """

    def __init__(
        self,
        store: MemoryStore,
        embed_api_key: str,
        embed_base_url: str,
        embed_model: str = "text-embedding-v3",
        fast_llm: LLMClient | None = None,
    ):
        """
        参数:
            store:         已初始化的 MemoryStore 实例
            embed_api_key: embedding API 的 Key
            embed_base_url: embedding API 的地址
            embed_model:   embedding 模型名
            fast_llm:      用于事实萃取的轻量 LLM（推荐用 qwen-flash）
                           None 表示不启用自动萃取
        """
        self._store = store
        self._embed_model = embed_model
        self._fast_llm = fast_llm

        self._embed_client = AsyncOpenAI(
            api_key=embed_api_key,
            base_url=embed_base_url,
        )

    async def initialize(self) -> None:
        """初始化（目前只是确保 store 已就绪）"""
        logger.info(
            "记忆服务就绪: embed_model=%s, fast_llm=%s",
            self._embed_model,
            self._fast_llm.model if self._fast_llm else "None",
        )

    @property
    def store(self) -> MemoryStore:
        """暴露底层 store，允许直接调 store 的方法"""
        return self._store

    # ── Embedding ─────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """将文本转为向量

        调用 embedding API，返回 float 列表。
        """
        if not text.strip():
            return []

        try:
            resp = await self._embed_client.embeddings.create(
                model=self._embed_model,
                input=text,
            )
            vector = resp.data[0].embedding
            logger.debug("embedding 完成: text=%s… dim=%d", text[:30], len(vector))
            return vector
        except Exception as e:
            logger.exception("embedding 调用失败")
            return []

    # ── 写入 ───────────────────────────────────────────

    async def add_memory(
        self,
        content: str,
        memory_type: str = "fact",
        user_id: str = "default",
        scope: str = "user",
        source: str | None = None,
        importance: float = 1.0,
        metadata: dict | None = None,
        dedup_threshold: float = 0.92,
    ) -> int:
        """添加一条记忆（自动 embedding + 去重检查）

        流程：
            1. 计算 content 的 embedding
            2. 检查数据库中是否存在相似内容（余弦相似度 > dedup_threshold）
            3. 如果存在，跳过写入，返回已有记忆的 id
            4. 如果不存在，写入新记忆

        参数:
            content:         记忆文本
            memory_type:     类型：fact | preference | event | profile
            user_id:         用户标识
            scope:           user / session / agent
            source:          来源
            importance:      重要性 [0~1]
            metadata:        额外数据
            dedup_threshold: 去重阈值（0~1，越高越严格）
        """
        embedding = await self.embed(content)
        if not embedding:
            logger.warning("embedding 失败，记忆 '%s' 不会被去重检查", content[:30])
            return await self._store.add(
                content=content, embedding=embedding, memory_type=memory_type,
                user_id=user_id, scope=scope, source=source,
                importance=importance, metadata=metadata,
            )

        # 去重检查：按类型+语义找相似
        similar = await self._store.find_similar(
            query_embedding=embedding,
            threshold=dedup_threshold,
            user_id=user_id,
            memory_type=memory_type,
        )
        if similar:
            existing = similar[0][0]
            logger.info(
                "记忆去重: '%s' 与已有记忆(id=%d) 相似度 %.3f，跳过写入",
                content[:40], existing.id, similar[0][1],
            )
            # 提升旧记忆的重要性 + 更新时间戳
            new_imp = min(existing.importance + 0.05, 1.0)
            await self._store.update_importance(existing.id, new_imp)
            return existing.id

        return await self._store.add(
            content=content, embedding=embedding, memory_type=memory_type,
            user_id=user_id, scope=scope, source=source,
            importance=importance, metadata=metadata,
        )

    # ── 搜索 ───────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str = "",
        scope: str = "",
        memory_type: str = "",
        min_importance: float = 0.0,
        hotness_alpha: float = 0.15,
        score_threshold: float = 0.5,
    ) -> list[MemoryItem]:
        """搜索记忆：embed 问题 → 向量搜索 → 返回

        参数:
            query:           搜索文本
            top_k:           返回条数
            user_id:         限定用户
            scope:           限定范围
            memory_type:     限定类型
            min_importance:  最小重要性
            hotness_alpha:   热度权重（0=纯语义，1=纯热度）
            score_threshold: 最低语义相似度阈值
        """
        query_emb = await self.embed(query)
        if not query_emb:
            return []
        return await self._store.search(
            query_embedding=query_emb,
            top_k=top_k,
            user_id=user_id,
            scope=scope,
            memory_type=memory_type,
            min_importance=min_importance,
            hotness_alpha=hotness_alpha,
            score_threshold=score_threshold,
            keyword_fallback=True,
        )

    # ── 事实萃取 ───────────────────────────────────────

    async def extract_facts(
        self,
        user_message: str,
        assistant_message: str,  # 保留参数但不再传给 LLM
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

    # ── 上下文构建 ─────────────────────────────────────

    async def build_memory_context(
        self,
        user_message: str,
        user_id: str = "default",
        max_memories: int = 3,
    ) -> str:
        """构建记忆上下文块

        只注入与当前话题高度相关的记忆，不相关的内容不注入。
        """
        memories = await self.search(
            query=user_message,
            top_k=max_memories,
            user_id=user_id,
            min_importance=0.5,
            hotness_alpha=0.0,
        )
        if not memories:
            return ""

        lines = []
        for m in memories:
            lines.append(f"- {m.content}")

        return MEMORY_CONTEXT_TEMPLATE.format(
            memories="\n".join(lines),
        ).strip()

    # ── 维护 ───────────────────────────────────────────

    async def cleanup_old_memories(self, before_days: int = 30) -> int:
        """清理旧的低重要性记忆"""
        return await self._store.delete_old(before_days)


# ── 便捷工厂 ──────────────────────────────────────────────

def create_memory_service(
    store: MemoryStore,
    embed_api_key: str,
    embed_base_url: str,
    embed_model: str = "text-embedding-v3",
    fast_llm_config: dict | None = None,
) -> MemoryService:
    """创建 MemoryService 实例

    fast_llm_config 示例：
        {
            "api_key": "sk-xxx",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-flash",
        }
    """
    fast_llm = None
    if fast_llm_config:
        fast_llm = LLMClient(
            api_key=fast_llm_config["api_key"],
            base_url=fast_llm_config["base_url"],
            model=fast_llm_config["model"],
        )

    return MemoryService(
        store=store,
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        fast_llm=fast_llm,
    )
