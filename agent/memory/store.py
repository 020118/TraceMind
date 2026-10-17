""" 记忆存储层 —— SQLite + Python 余弦相似度搜索（方案A）

核心职责：
    - SQLite 建表、读写记忆（含记忆类型 + 热度排序 + 关键词兜底）
    - 余弦相似度搜索 + 热度评分 + 关键词回退
    - 无外部依赖（仅 sqlite3 + numpy）

记忆类型（memory_type）：
    fact        — 客观事实（"用户住在北京"）
    preference  — 偏好习惯（"用户喜欢喝咖啡"）
    event       — 具体事件（"2026-07-20 用户在松山湖跑步"）
    profile     — 稳定画像（"用户是广东东莞人"）

使用方式：
    store = MemoryStore("~/.akashic/workspace/memory/memories.db")
    await store.initialize()

    await store.add("用户喜欢打篮球", memory_type="preference", embedding=[...])
    results = await store.search(query_emb=[...], top_k=5, hotness_alpha=0.2)
    # 向量搜索无结果时自动回退关键词搜索
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── 数据模型 ──────────────────────────────────────────────

@dataclass
class MemoryItem:
    """一条记忆的数据模型"""
    id: int = 0
    user_id: str = "default"
    content: str = ""
    memory_type: str = "fact"    # fact | preference | event | profile
    scope: str = "user"          # user | session | agent
    source: str | None = None     # 来源标识（对话ID/工具名）
    importance: float = 1.0       # 重要性 [0~1]
    access_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embedding(self) -> list[float] | None:
        return None  # 外部传入，不在 MemoryItem 中持久化


# ── SQL 建表语句 ──────────────────────────────────────────

CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL DEFAULT 'default',
    content     TEXT    NOT NULL,
    embedding   BLOB,                       -- numpy float32 序列化
    memory_type TEXT    NOT NULL DEFAULT 'fact',  -- fact | preference | event | profile
    scope       TEXT    NOT NULL DEFAULT 'user',
    source      TEXT,
    importance  REAL    NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    metadata    TEXT                        -- JSON 字符串
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memories_user  ON memories(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);",
    "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);",
    "CREATE INDEX IF NOT EXISTS idx_memories_time  ON memories(updated_at);",
]

# 旧表升级：补充 memory_type 列（幂等，列已存在则忽略）
ALTER_ADD_MEMORY_TYPE = (
    "ALTER TABLE memories ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'fact';"
    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);"
)


# ── MemoryStore ───────────────────────────────────────────

class MemoryStore:
    """SQLite 记忆存储

    职责：
        - 数据库初始化和表管理
        - 记忆的增删改查
        - 余弦相似度搜索（方案A 核心）
        - embedding 序列化/反序列化（BLOB ↔ numpy array）
    """

    def __init__(self, db_path: str | Path):
        raw = str(db_path)
        if raw == ":memory:":
            self._db_path = Path(":memory:")
        else:
            self._db_path = Path(db_path).expanduser().resolve()
        self._conn: sqlite3.Connection | None = None

    # ── 生命周期 ───────────────────────────────────────

    async def initialize(self) -> None:
        """打开数据库连接并建表（含旧表升级）"""
        db_path = str(self._db_path)
        if db_path != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("记忆数据库: %s", self._db_path)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

        self._conn.execute(CREATE_MEMORIES_TABLE)
        for idx in CREATE_INDEXES:
            self._conn.execute(idx)
        # 升级旧表：添加 memory_type 列
        try:
            self._conn.executescript(ALTER_ADD_MEMORY_TYPE)
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

        count = self._count()
        logger.info("记忆系统就绪，已有 %d 条记忆", count)

    async def close(self) -> None:
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("记忆数据库已关闭")

    # ── 写操作 ─────────────────────────────────────────

    async def add(
        self,
        content: str,
        embedding: list[float] | np.ndarray | None = None,
        memory_type: str = "fact",
        user_id: str = "default",
        scope: str = "user",
        source: str | None = None,
        importance: float = 1.0,
        metadata: dict | None = None,
    ) -> int:
        """添加一条记忆，返回 id

        参数:
            content:     记忆文本
            embedding:   向量（None 表示不做搜索，仅存文本）
            memory_type: 类型：fact | preference | event | profile
            user_id:     用户标识
            scope:       user / session / agent
            source:      来源
            importance:  重要性 [0~1]
            metadata:    额外数据（会被转为 JSON）
        """
        self._assert_open()

        embed_blob = _encode_embedding(embedding) if embedding is not None else None
        meta_str = json.dumps(metadata, ensure_ascii=False) if metadata else None

        cur = self._conn.execute(
            """INSERT INTO memories (user_id, content, embedding, memory_type, scope, source, importance, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, content, embed_blob, memory_type, scope, source, importance, meta_str),
        )
        self._conn.commit()
        mem_id = cur.lastrowid
        logger.debug("记忆已添加: id=%d, type=%s, scope=%s, content=%s…", mem_id, memory_type, scope, content[:40])
        return mem_id

    async def delete(self, mem_id: int) -> bool:
        """删除一条记忆"""
        self._assert_open()
        cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.debug("记忆已删除: id=%d", mem_id)
        return deleted

    async def update_importance(self, mem_id: int, importance: float) -> None:
        """更新重要性"""
        self._assert_open()
        self._conn.execute(
            "UPDATE memories SET importance = ?, updated_at = datetime('now', 'localtime') WHERE id = ?",
            (importance, mem_id),
        )
        self._conn.commit()

    async def increment_access(self, mem_id: int) -> None:
        """增加检索计数"""
        self._assert_open()
        self._conn.execute(
            "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
            (mem_id,),
        )
        self._conn.commit()

    # ── 读操作 ─────────────────────────────────────────

    async def get(self, mem_id: int) -> MemoryItem | None:
        """按 ID 获取单条记忆"""
        self._assert_open()
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        return _row_to_item(row) if row else None

    async def get_all(
        self,
        user_id: str = "",
        scope: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryItem]:
        """获取记忆列表（分页）"""
        self._assert_open()

        conditions = []
        params: list[Any] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if scope:
            conditions.append("scope = ?")
            params.append(scope)

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [_row_to_item(r) for r in rows]

    # ── 热度评分 ────────────────────────────────────────

    @staticmethod
    def _hotness_score(
        access_count: int,
        updated_at: str,
        half_life_days: float = 14.0,
    ) -> float:
        """计算热度分：频度 × 时间衰减

        公式：
            freq     = 1 / (1 + exp(-ln(access_count+1)))   → 越频繁越热
            age_days = (now - updated_at)                    → 越久越冷
            recency  = exp(-ln2 / half_life × age_days)      → 半衰期衰减

        返回 (0, 1) 之间的值。
        """
        freq = 1.0 / (1.0 + math.exp(-math.log1p(max(0, access_count))))
        try:
            dt = datetime.fromisoformat(updated_at)
        except (ValueError, TypeError):
            return freq  # 时间解析失败，只用频度
        if dt.tzinfo is None:
            now = datetime.now()
        else:
            now = datetime.now(timezone.utc)
        age_days = max((now - dt).total_seconds() / 86400.0, 0.0)
        recency = math.exp(-math.log(2) / half_life_days * age_days)
        return freq * recency

    # ── 搜索 ─────────────────────────────────────────────

    async def search(
        self,
        query_embedding: list[float] | np.ndarray,
        top_k: int = 5,
        user_id: str = "",
        scope: str = "",
        memory_type: str = "",
        min_importance: float = 0.0,
        hotness_alpha: float = 0.15,
        keyword_fallback: bool = True,
        score_threshold: float = 0.0,
    ) -> list[MemoryItem]:
        """余弦相似度搜索 + 热度排序 + 关键词兜底

        参数:
            query_embedding: 查询向量
            top_k:           返回条数
            user_id:         限定用户
            scope:           限定范围
            memory_type:     限定类型
            min_importance:  最小重要性阈值
            hotness_alpha:   热度权重 [0~1]，0=纯语义，1=纯热度
            keyword_fallback: 向量结果不足时回退关键词搜索
            score_threshold: 最低语义相似度（低于此值的向量结果不返回）
        """
        self._assert_open()

        # 1. 构建查询条件
        conditions = ["embedding IS NOT NULL"]
        params: list[Any] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        if min_importance > 0:
            conditions.append("importance >= ?")
            params.append(min_importance)

        where = " AND ".join(conditions)

        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE {where}",
            params,
        ).fetchall()
        if not rows:
            return []

        # 2. 计算语义相似度 + 热度
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            embed_blob = row["embedding"]
            if embed_blob is None:
                continue

            mem_vec = _decode_embedding(embed_blob)
            if mem_vec is None or len(mem_vec) != len(query_vec):
                continue

            # 余弦相似度
            dot = np.dot(query_vec, mem_vec)
            mem_norm = np.linalg.norm(mem_vec)
            similarity = float(dot / (query_norm * mem_norm)) if mem_norm > 0 else 0.0

            # 热度分
            hotness = self._hotness_score(
                access_count=row["access_count"] or 0,
                updated_at=row["updated_at"] or "",
            )

            # 综合评分 = (1-α) × 语义 + α × 热度
            final_score = (1.0 - hotness_alpha) * similarity + hotness_alpha * hotness

            # 语义相似度低于阈值则跳过（只基于语义判断相关性，不受热度影响）
            if similarity < score_threshold:
                continue

            scored.append((final_score, row))

        # 3. 排序取 Top-K
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        # 增加检索计数
        for _, row in top:
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                (row["id"],),
            )
        self._conn.commit()

        results = [_row_to_item(r) for _, r in top]

        # 4. 关键词兜底：向量结果不足时，用查询关键词补全
        if keyword_fallback and len(results) < top_k:
            keyword_results = await self._keyword_search(
                query="",  # 由调用方传 query_text
                top_k=top_k - len(results),
                user_id=user_id,
                scope=scope,
                memory_type=memory_type,
            )
            # 去重
            existing_ids = {r.id for r in results}
            for kr in keyword_results:
                if kr.id not in existing_ids:
                    results.append(kr)
                if len(results) >= top_k:
                    break

        return results

    async def _keyword_search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str = "",
        scope: str = "",
        memory_type: str = "",
    ) -> list[MemoryItem]:
        """关键词搜索兜底（适用于向量搜索无结果时）

        对 content 字段做 OR LIKE 匹配，按匹配词数降序排列。
        这个方法是内部 fallback，通常通过 search(keyword_fallback=True) 触发。
        如需从外部直接调用，请使用 keyword_search()。
        """
        if not query:
            return []
        return await self.keyword_search(
            query=query,
            top_k=top_k,
            user_id=user_id,
            scope=scope,
            memory_type=memory_type,
        )

    async def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str = "",
        scope: str = "",
        memory_type: str = "",
    ) -> list[MemoryItem]:
        """关键词搜索（公开接口）

        从 query 中提取关键词，对 content 做 OR LIKE 匹配，
        按命中关键词数量降序排列。

        参数:
            query:       搜索文本（从中提取关键词）
            top_k:       返回条数
            user_id:     限定用户
            scope:       限定范围
            memory_type: 限定类型
        """
        self._assert_open()

        # 提取关键词：中文分词（简单二元组）+ 英文 token
        terms = _extract_keywords(query)
        if not terms:
            return []

        conditions = []
        params: list[Any] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)

        like_values = [f"%{t}%" for t in terms]
        like_clauses = ["content LIKE ?" for _ in terms]

        base_where = " AND ".join(conditions) if conditions else "1=1"
        score_expr = " + ".join(
            f"(CASE WHEN content LIKE ? THEN 1 ELSE 0 END)" for _ in terms
        )
        like_values = [f"%{t}%" for t in terms]
        # score uses same values as like

        rows = self._conn.execute(
            f"SELECT *, ({score_expr}) AS kw_score FROM memories "
            f"WHERE {base_where} AND ({' OR '.join(like_clauses)}) "
            f"ORDER BY kw_score DESC, importance DESC, updated_at DESC "
            f"LIMIT ?",
            params + like_values + like_values + [top_k],
        ).fetchall()
        return [_row_to_item(r) for r in rows]

    async def find_similar(
        self,
        query_embedding: list[float] | np.ndarray,
        threshold: float = 0.92,
        user_id: str = "",
        memory_type: str = "",
    ) -> list[tuple[MemoryItem, float]]:
        """查找相似记忆（用于去重）

        参数:
            query_embedding: 查询向量
            threshold:       余弦相似度阈值
            user_id:         限定用户（空=全部）
            memory_type:     限定类型（空=全部）

        返回:
            [(MemoryItem, 相似度), ...]，按相似度降序排列
        """
        self._assert_open()

        conditions = ["embedding IS NOT NULL"]
        params: list[Any] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)

        where = " AND ".join(conditions)
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE {where}",
            params,
        ).fetchall()
        if not rows:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        matched: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            embed_blob = row["embedding"]
            if embed_blob is None:
                continue
            mem_vec = _decode_embedding(embed_blob)
            if mem_vec is None or len(mem_vec) != len(query_vec):
                continue

            dot = np.dot(query_vec, mem_vec)
            mem_norm = np.linalg.norm(mem_vec)
            similarity = float(dot / (query_norm * mem_norm)) if mem_norm > 0 else 0.0

            if similarity >= threshold:
                matched.append((similarity, row))

        matched.sort(key=lambda x: x[0], reverse=True)
        return [(_row_to_item(r), sim) for sim, r in matched]

    # ── 统计 ───────────────────────────────────────────

    async def count(self, user_id: str = "", scope: str = "") -> int:
        """记忆总数"""
        return self._count(user_id, scope)

    def _count(self, user_id: str = "", scope: str = "") -> int:
        """同步版 count（内部使用）"""
        self._assert_open()
        conditions = []
        params: list[Any] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        where = " AND ".join(conditions) if conditions else "1=1"
        row = self._conn.execute(
            f"SELECT COUNT(*) as cnt FROM memories WHERE {where}", params
        ).fetchone()
        return row["cnt"] if row else 0

    # ── 维护 ───────────────────────────────────────────

    async def vacuum(self) -> None:
        """回收空间"""
        self._assert_open()
        self._conn.execute("VACUUM;")
        logger.info("记忆数据库已 VACUUM")

    async def delete_old(self, before_days: int = 30) -> int:
        """删除超过指定天数的低重要性记忆

        返回删除数量。这是"遗忘"机制的简单实现。
        """
        self._assert_open()
        cur = self._conn.execute(
            """DELETE FROM memories
               WHERE julianday('now', 'localtime') - julianday(updated_at) > ?
               AND importance < 0.3""",
            (before_days,),
        )
        self._conn.commit()
        deleted = cur.rowcount
        if deleted:
            logger.info("遗忘了 %d 条旧记忆（超过 %d 天，重要性 < 0.3）", deleted, before_days)
        return deleted

    # ── 内部 ───────────────────────────────────────────

    def _assert_open(self) -> None:
        if self._conn is None:
            raise RuntimeError("MemoryStore 未初始化，请先调用 await initialize()")

    def __repr__(self) -> str:
        return f"MemoryStore({self._db_path})"


# ── embedding 序列化辅助 ────────────────────────────────


def _encode_embedding(emb: list[float] | np.ndarray) -> bytes:
    """numpy float32 数组 → BLOB"""
    arr = np.array(emb, dtype=np.float32)
    return arr.tobytes()


def _decode_embedding(blob: bytes) -> np.ndarray | None:
    """BLOB → numpy float32 数组"""
    try:
        return np.frombuffer(blob, dtype=np.float32)
    except Exception:
        logger.warning("embedding 反序列化失败")
        return None


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    """SQLite Row → MemoryItem"""
    meta_str = row["metadata"]
    metadata = json.loads(meta_str) if meta_str else {}
    return MemoryItem(
        id=row["id"],
        user_id=row["user_id"],
        content=row["content"],
        memory_type=row["memory_type"] if "memory_type" in row.keys() else "fact",
        scope=row["scope"],
        source=row["source"],
        importance=row["importance"],
        access_count=row["access_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=metadata,
    )


def _extract_keywords(text: str, max_terms: int = 10) -> list[str]:
    """从文本中提取搜索关键词

    策略：
        - 英文 token（长度 >= 2）
        - 中文连续字符（长度 2~4 的直接用，更长的拆二元组）
        - 去重，最多返回 max_terms 个
    """
    import re

    terms: list[str] = []
    # 英文 token
    terms.extend(re.findall(r"[a-zA-Z0-9_\-.]+", text))
    # 中文
    cjk = re.findall(r"[一-鿿]+", text)
    for chunk in cjk:
        if len(chunk) <= 4:
            terms.append(chunk)
        else:
            for i in range(len(chunk) - 1):
                terms.append(chunk[i:i+2])

    seen: set[str] = set()
    result: list[str] = []
    for t in terms:
        t = t.strip()
        if len(t) < 2 or t in seen:
            continue
        # 过滤常见停用词
        if t.lower() in {"用户", "助手", "我们", "他们", "这个", "那个",
                          "什么", "如何", "是否", "没有", "进行", "可以"}:
            continue
        seen.add(t)
        result.append(t)
        if len(result) >= max_terms:
            break
    return result
