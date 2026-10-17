""" 六层记忆存储层 —— Markdown 文件 + SQLite 幂等索引 + 两阶段提交

核心职责：
    - 六层 Markdown 文件存储（MEMORY/SELF/PENDING/HISTORY/RECENT_CONTEXT/journal）
    - SQLite 幂等索引 + 文件尾部标记双重去重
    - 两阶段提交 Consolidation（PENDING → MEMORY），原子快照 + 回滚
    - 无外部依赖（仅 sqlite3 + 标准库）
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 合并标记前缀/后缀
_CONSOLIDATION_MARKER_PREFIX = "<!-- consolidation:"
_CONSOLIDATION_MARKER_SUFFIX = " -->"
# 扫描文件尾部字节数用于检测标记
_CONSOLIDATION_TAIL_BYTES = 1024 * 1024  # 1MB


@dataclass
class MemoryStoreConfig:
    """存储配置"""
    workspace: Path
    max_history_chars: int = 100000  # 单会话历史最大字符数
    consolidation_interval_seconds: int = 3600  # Consolidation 间隔（秒）


class MemoryStoreV2:
    """六层记忆存储 V2

    六层架构：
        - MEMORY.md   : 稳定用户画像，仅 Optimizer 写入
        - SELF.md     : Agent 自我认知，Optimizer 更新
        - PENDING.md  : 增量事实缓冲，对话中高频追加
        - HISTORY.md  : 永久事件日志，幂等追加
        - RECENT_CONTEXT.md : 紧凑近期摘要，供 Proactive/Drift 用
        - journal/    : 按天时间线，append-only YYYY-MM-DD.md

    核心机制：
        - 幂等写入：SQLite 索引 + 文件尾部标记双重去重
        - 两阶段提交 Consolidation：原子快照 + 回滚
        - 崩溃自愈：启动自动恢复
    """

    def __init__(self, config: MemoryStoreConfig):
        self.config = config
        self.memory_dir = config.workspace / "memory"
        self.journal_dir = self.memory_dir / "journal"

        # 文件路径
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.self_file = self.memory_dir / "SELF.md"
        self.pending_file = self.memory_dir / "PENDING.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.recent_context_file = self.memory_dir / "RECENT_CONTEXT.md"
        self.pending_file = self.memory_dir / "PENDING.md"
        self._consolidation_db = self.memory_dir / "consolidation_writes.db"
        self._consolidation_lock = threading.Lock()

        # 快照文件路径
        self._snapshot_path = self.pending_file.with_name("PENDING.snapshot.md")

        # 会话历史缓存（内存中按 session_key 隔离）
        self._session_histories: dict[str, list[dict]] = {}

    # ── 生命周期 ───────────────────────────────────────

    async def initialize(self) -> None:
        """初始化目录、文件、数据库"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.journal_dir.mkdir(parents=True, exist_ok=True)

        # 确保文件存在
        for f in [self.memory_file, self.self_file, self.pending_file,
                  self.history_file, self.recent_context_file, self.pending_file]:
            if not f.exists():
                f.write_text("", encoding="utf-8")

        # 初始化 SQLite 幂等索引
        self._init_consolidation_db()

        # 崩溃恢复：处理遗留 snapshot
        self._recover_pending_snapshot()

        logger.info("记忆存储 V2 初始化完成: %s", self.memory_dir)

    def _init_consolidation_db(self) -> None:
        """初始化幂等索引数据库"""
        conn = sqlite3.connect(str(self._consolidation_db))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_writes (
                    source_ref TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT,
                    trailing_blank_line INTEGER NOT NULL DEFAULT 0,
                    done_at TEXT NOT NULL,
                    PRIMARY KEY (source_ref, kind)
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _recover_pending_snapshot(self) -> None:
        """启动时检测遗留 snapshot 并自动回滚"""
        if self._snapshot_path.exists():
            logger.warning("[memory] 检测到遗留 PENDING.snapshot.md，执行崩溃回滚")
            self.rollback_pending_snapshot()

    # ── 核心：幂等写入 ───────────────────────────────────

    def _consolidation_marker(self, source_ref: str, kind: str) -> str:
        """生成合并标记"""
        src = (source_ref or "").replace("\n", " ").strip()
        kd = (kind or "").replace("\n", " ").strip()
        return f"{_CONSOLIDATION_MARKER_PREFIX}{src}:{kd}{_CONSOLIDATION_MARKER_SUFFIX}"

    @staticmethod
    def _strip_consolidation_markers(text: str) -> str:
        """剥离 consolidation 标记行"""
        lines = text.splitlines()
        kept = [
            line for line in lines
            if not (line.startswith(_CONSOLIDATION_MARKER_PREFIX)
                    and line.endswith(_CONSOLIDATION_MARKER_SUFFIX))
        ]
        return "\n".join(kept).strip()

    def _tail_contains_marker(self, path: Path, marker: str) -> bool:
        """检查文件尾部是否包含标记（用于崩溃恢复）"""
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                take = min(size, _CONSOLIDATION_TAIL_BYTES)
                if take <= 0:
                    return False
                f.seek(size - take)
                tail = f.read(take).decode("utf-8", errors="ignore")
                return marker in tail
        except Exception:
            return False

    def _file_contains_marker(self, path: Path, marker: str) -> bool:
        """检查文件是否包含标记（全文扫描）"""
        if not path.exists():
            return False
        needle = marker.encode("utf-8")
        if not needle:
            return False
        carry = b""
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    data = carry + chunk
                    if needle in data:
                        return True
                    if len(needle) > 1:
                        carry = data[-(len(needle) - 1):]
                    else:
                        carry = b""
        except Exception:
            return False
        return False

    def _append_once_with_index(
        self,
        *,
        target_file: Path,
        text: str,
        source_ref: str,
        kind: str,
        trailing_blank_line: bool = False,
    ) -> bool:
        """幂等追加：仅当 (source_ref, kind) 未写入时才写入

        双重去重：SQLite 索引 + 文件尾部标记
        返回 True 表示首次写入，False 表示已存在（幂等）
        """
        marker = self._consolidation_marker(source_ref, kind)
        src = (source_ref or "").strip()
        kd = (kind or "").strip()
        if not src or not kd or not text:
            return False

        with self._consolidation_lock:
            conn = sqlite3.connect(str(self._consolidation_db), timeout=30.0)
            try:
                conn.execute("BEGIN IMMEDIATE")

                # 1. 查 SQLite 索引
                row = conn.execute(
                    "SELECT payload, trailing_blank_line FROM consolidation_writes WHERE source_ref=? AND kind=?",
                    (source_ref, kind),
                ).fetchone()

                if row is not None:
                    existing_payload = row[0] or ""
                    existing_trailing = bool(int(row[1] or 0))
                    # 索引有，但文件无标记 → 文件写漏了，补写
                    if not self._file_contains_marker(target_file, self._consolidation_marker(source_ref, kind)):
                        if existing_payload:
                            with open(target_file, "a", encoding="utf-8") as f:
                                f.write(self._consolidation_marker(source_ref, kind) + "\n")
                                f.write(existing_payload.rstrip() + "\n")
                                if existing_trailing:
                                    f.write("\n")
                        conn.execute("COMMIT")
                        return False  # 已存在

                # 2. 索引无，但文件尾部有标记 → 索引写漏了，补索引
                if self._tail_contains_marker(target_file, self._consolidation_marker(source_ref, kind)):
                    conn.execute(
                        "INSERT OR REPLACE INTO consolidation_writes(source_ref, kind, payload, trailing_blank_line, done_at) VALUES (?, ?, ?, ?, datetime('now'))",
                        (source_ref, kind, text, 1 if trailing_blank_line else 0),
                    )
                    conn.execute("COMMIT")
                    return False

                # 3. 正常路径：写文件 + 写索引
                marker = self._consolidation_marker(source_ref, kind)
                with open(target_file, "a", encoding="utf-8") as f:
                    f.write(marker + "\n")
                    f.write(text.rstrip() + "\n")
                    if trailing_blank_line:
                        f.write("\n")

                conn.execute(
                    "INSERT OR REPLACE INTO consolidation_writes(source_ref, kind, payload, trailing_blank_line, done_at) VALUES (?, ?, ?, ?, datetime('now'))",
                    (source_ref, kind, text, 1 if trailing_blank_line else 0),
                )
                conn.execute("COMMIT")
                return True

            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    # ── 对外写入接口 ─────────────────────────────────────

    def append_pending(self, text: str, source_ref: str) -> bool:
        """追加增量事实到 PENDING.md（幂等）"""
        if not text or not text.strip():
            return False
        return self._append_once_with_index(
            target_file=self.pending_file,
            text=text.strip(),
            source_ref=source_ref,
            kind="pending",
            trailing_blank_line=False,
        )

    def append_history_once(self, text: str, source_ref: str, kind: str = "history_entry") -> bool:
        """幂等追加到 HISTORY.md"""
        if not text or not text.strip():
            return False
        return self._append_once_with_index(
            target_file=self.history_file,
            text=text.strip(),
            source_ref=source_ref,
            kind=kind,
            trailing_blank_line=True,
        )

    def append_journal(self, date_str: str, text: str, source_ref: str, kind: str = "journal") -> bool:
        """追加到 journal/YYYY-MM-DD.md（幂等）"""
        if not text or not text.strip():
            return False
        journal_file = self.journal_dir / f"{date_str}.md"
        if not journal_file.exists():
            journal_file.write_text(f"# {date_str}\n\n", encoding="utf-8")
        return self._append_once_with_index(
            target_file=journal_file,
            text=text.strip(),
            source_ref=source_ref,
            kind=kind,
            trailing_blank_line=True,
        )

    # ── 读取接口 ────────────────────────────────────────

    def read_long_term(self) -> str:
        """读取长期记忆 (MEMORY.md)"""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        """写入长期记忆（仅 Optimizer 调用）"""
        self.memory_file.write_text(content, encoding="utf-8")

    def read_self(self) -> str:
        if self.self_file.exists():
            return self.self_file.read_text(encoding="utf-8")
        return ""

    def write_self(self, content: str) -> None:
        self.self_file.write_text(content, encoding="utf-8")

    def read_recent_context(self) -> str:
        if self.recent_context_file.exists():
            return self.recent_context_file.read_text(encoding="utf-8")
        return ""

    def write_recent_context(self, content: str) -> None:
        self.recent_context_file.write_text(content, encoding="utf-8")

    def read_history(self, max_chars: int = 0) -> str:
        if not self.history_file.exists():
            return ""
        text = self.history_file.read_text(encoding="utf-8")
        text = self._strip_consolidation_markers(text)
        if max_chars > 0 and len(text) > max_chars:
            return text[-max_chars:]
        return text

    def read_pending(self) -> str:
        if self.pending_file.exists():
            return self._strip_consolidation_markers(
                self.pending_file.read_text(encoding="utf-8")
            )
        return ""

    def get_memory_context(self) -> str:
        """构建注入 prompt 的记忆上下文（长期记忆 + 增量待归档事实）"""
        parts = []
        long_term = self.read_long_term()
        if long_term:
            parts.append(f"## Long-term Memory\n{long_term}")
        pending = self.read_pending()
        if pending:
            parts.append(f"## Pending Facts (Newly Extracted, Not Yet Consolidated)\n{pending}")
        return "\n\n".join(parts) if parts else ""

    # ── 两阶段提交 Consolidation ────────────────────────

    async def snapshot_pending(self) -> str:
        """Phase-1: 原子移走 PENDING.md，返回其内容

        rename 之后 append_pending 会写入新建的 PENDING.md，
        与本次快照完全隔离，不会丢失后续增量。
        """
        self._recover_pending_snapshot()
        if not self.pending_file.exists() or self.pending_file.stat().st_size == 0:
            return ""
        # POSIX rename 是原子操作：rename 完成后新追加写入全新的 PENDING.md
        snapshot_path = self.pending_file.with_name("PENDING.snapshot.md")
        self.pending_file.rename(snapshot_path)
        return self._strip_consolidation_markers(
            snapshot_path.read_text(encoding="utf-8")
        )

    def commit_pending_snapshot(self) -> None:
        """Phase-2 成功：merge 已完成，删除快照"""
        snapshot_path = self.pending_file.with_name("PENDING.snapshot.md")
        if snapshot_path.exists():
            snapshot_path.unlink()
        # 保持 PENDING.md 常驻，避免"已归档后文件消失"带来的状态歧义
        if not self.pending_file.exists():
            self.pending_file.touch()

    def rollback_pending_snapshot(self) -> None:
        """Phase-2 失败：将快照内容合并回 PENDING.md，不丢失任何数据

        快照（较旧）在前，运行期新追加（较新）在后。
        """
        snapshot_path = self.pending_file.with_name("PENDING.snapshot.md")
        if not snapshot_path.exists():
            return
        snap_text = snapshot_path.read_text(encoding="utf-8")
        new_text = (
            self.pending_file.read_text(encoding="utf-8")
            if self.pending_file.exists()
            else ""
        )
        merged = snap_text.rstrip() + "\n" + new_text if new_text.strip() else snap_text
        self.pending_file.write_text(merged, encoding="utf-8")
        snapshot_path.unlink()
        logger.info("[memory] PENDING snapshot 已回滚合并")

    def _recover_pending_snapshot(self) -> None:
        """启动时或 snapshot_pending 前调用，处理上次崩溃遗留的快照"""
        snapshot_path = self.pending_file.with_name("PENDING.snapshot.md")
        if snapshot_path.exists():
            logger.warning("[memory] 检测到遗留 PENDING.snapshot.md，执行崩溃回滚")
            self.rollback_pending_snapshot()

    def commit_pending_snapshot(self) -> None:
        """Phase-2 成功：merge 已完成，删除快照"""
        snapshot_path = self.pending_file.with_name("PENDING.snapshot.md")
        if snapshot_path.exists():
            snapshot_path.unlink()
        # 保持 PENDING.md 常驻，避免"已归档后文件消失"带来的状态歧义
        if not self.pending_file.exists():
            self.pending_file.touch()

    def rollback_pending_snapshot(self) -> None:
        """Phase-2 失败：将快照内容合并回 PENDING.md，不丢失任何数据

        快照（较旧）在前，运行期新追加（较新）在后。
        """
        snapshot_path = self.pending_file.with_name("PENDING.snapshot.md")
        if not snapshot_path.exists():
            return
        snap_text = snapshot_path.read_text(encoding="utf-8")
        new_text = (
            self.pending_file.read_text(encoding="utf-8")
            if self.pending_file.exists()
            else ""
        )
        merged = snap_text.rstrip() + "\n" + new_text if new_text.strip() else snap_text
        self.pending_file.write_text(merged, encoding="utf-8")
        snapshot_path.unlink()
        logger.info("[memory] PENDING snapshot 已回滚合并")

    def _recover_pending_snapshot(self) -> None:
        """启动时或 snapshot_pending 前调用，处理上次崩溃遗留的快照"""
        snapshot_path = self.pending_file.with_name("PENDING.snapshot.md")
        if snapshot_path.exists():
            logger.warning("[memory] 检测到遗留 PENDING.snapshot.md，执行崩溃回滚")
            self.rollback_pending_snapshot()

    # ── 会话历史隔离 ────────────────────────────────────

    def get_session_history(self, session_key: str) -> list[dict]:
        """获取会话历史（不存在则自动创建空列表）"""
        if session_key not in self._session_histories:
            self._session_histories[session_key] = []
        return self._session_histories[session_key]

    def append_session_history(self, session_key: str, role: str, content: str) -> None:
        """追加一条消息到会话历史（带截断）"""
        history = self.get_session_history(session_key)
        history.append({"role": role, "content": content})
        # 截断：保留最近配置字符数
        total_chars = sum(len(m.get("content", "")) for m in history)
        while total_chars > self.config.max_history_chars and len(history) >= 2:
            removed = history.pop(0)
            total_chars -= len(removed.get("content", ""))
            if history:
                removed = history.pop(0)
                total_chars -= len(removed.get("content", ""))

    def get_session_history_for_prompt(self, session_key: str) -> list[dict]:
        """获取用于 prompt 的会话历史（浅拷贝）"""
        return list(self.get_session_history(session_key))

    # ── 维护 ────────────────────────────────────────────

    async def close(self) -> None:
        """关闭（无持久连接，无需操作）"""
        logger.info("记忆存储 V2 已关闭")

    def __repr__(self) -> str:
        return f"MemoryStoreV2({self.memory_dir})"