"""Durable operation journal and outbound outbox.

The model may decide *what* to do, but this store records *where execution is*
before a destructive side effect happens.  SQLite uses WAL plus FULL
synchronous mode so a process restart can reconcile unfinished work.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
CONFIRMED = "CONFIRMED"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
NEEDS_REVIEW = "NEEDS_REVIEW"

def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class OperationRecord:
    operation_id: str
    session_key: str
    channel: str
    chat_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: str
    result: str
    error: str
    created_at: str
    updated_at: str
    expires_at: str | None


class ReliabilityStore:
    """Small SQLite store shared by the Agent loop and outbound dispatcher."""

    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path
        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tool_operations (
                    operation_id TEXT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    channel TEXT NOT NULL DEFAULT '',
                    chat_id TEXT NOT NULL DEFAULT '',
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_tool_operations_session_status
                ON tool_operations(session_key, status, created_at);

                CREATE TABLE IF NOT EXISTS outbound_outbox (
                    outbox_id TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    sent_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_outbound_outbox_status
                ON outbound_outbox(status, created_at);
                """
            )
            self._ensure_operation_columns()
            self._conn.commit()

    def _ensure_operation_columns(self) -> None:
        rows = self._conn.execute("PRAGMA table_info(tool_operations)").fetchall()
        existing = {str(row["name"]) for row in rows}
        if "channel" not in existing:
            self._conn.execute(
                "ALTER TABLE tool_operations ADD COLUMN channel TEXT NOT NULL DEFAULT ''"
            )
        if "chat_id" not in existing:
            self._conn.execute(
                "ALTER TABLE tool_operations ADD COLUMN chat_id TEXT NOT NULL DEFAULT ''"
            )

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            try:
                self.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()

    def create_pending_operation(
        self,
        *,
        session_key: str,
        channel: str = "",
        chat_id: str = "",
        tool_name: str,
        arguments: dict[str, Any],
        expires_in: timedelta = timedelta(minutes=30),
    ) -> OperationRecord:
        now = _now()
        expires_at = (datetime.now(UTC) + expires_in).isoformat()
        operation_id = uuid.uuid4().hex
        payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        with self._lock:
            # A session may have only one live confirmation.  Replacing it is
            # explicit and durable rather than silently overwriting a dict.
            self._conn.execute(
                """
                UPDATE tool_operations
                SET status = ?, error = ?, updated_at = ?
                WHERE session_key = ? AND status = ?
                """,
                (
                    CANCELLED,
                    "被同一会话中的新确认请求替换",
                    now,
                    session_key,
                    PENDING_CONFIRMATION,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO tool_operations (
                    operation_id, session_key, channel, chat_id,
                    tool_name, arguments_json,
                    status, created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    operation_id,
                    session_key,
                    channel,
                    chat_id,
                    tool_name,
                    payload,
                    PENDING_CONFIRMATION,
                    now,
                    now,
                    expires_at,
                ),
            )
            self._conn.commit()
        record = self.get_operation(operation_id)
        if record is None:  # pragma: no cover - SQLite invariant
            raise RuntimeError("创建工具操作日志失败")
        return record

    def finish_operation(
        self,
        operation_id: str,
        *,
        status: str,
        result: str,
        error: str = "",
        outbound_payload: dict[str, Any] | None = None,
    ) -> tuple[OperationRecord, str]:
        """Atomically finish an operation and reserve its durable reply."""

        now = _now()
        outbox_id = uuid.uuid4().hex if outbound_payload is not None else ""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                current = self._conn.execute(
                    "SELECT status FROM tool_operations WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                if current is None:
                    raise KeyError(f"未知操作: {operation_id}")
                if str(current["status"]) != RUNNING:
                    raise ValueError(
                        f"操作 {operation_id} 当前状态 {current['status']}，不能结束"
                    )
                self._conn.execute(
                    """
                    UPDATE tool_operations
                    SET status = ?, result = ?, error = ?, updated_at = ?
                    WHERE operation_id = ?
                    """,
                    (status, result, error, now, operation_id),
                )
                if outbound_payload is not None:
                    self._conn.execute(
                        """
                        INSERT INTO outbound_outbox (
                            outbox_id, channel, chat_id, payload_json,
                            status, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, 'PENDING', ?, ?)
                        """,
                        (
                            outbox_id,
                            str(outbound_payload.get("channel", "")),
                            str(outbound_payload.get("chat_id", "")),
                            json.dumps(
                                outbound_payload,
                                ensure_ascii=False,
                                default=str,
                            ),
                            now,
                            now,
                        ),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        record = self.get_operation(operation_id)
        if record is None:  # pragma: no cover - SQLite invariant
            raise RuntimeError("工具操作完成状态写入失败")
        return record, outbox_id

    def get_operation(self, operation_id: str) -> OperationRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tool_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        return self._operation_from_row(row) if row is not None else None

    def get_pending_operation(self, session_key: str) -> OperationRecord | None:
        now = _now()
        with self._lock:
            expired = self._conn.execute(
                """
                SELECT operation_id FROM tool_operations
                WHERE session_key = ? AND status = ?
                  AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (session_key, PENDING_CONFIRMATION, now),
            ).fetchall()
            if expired:
                self._conn.execute(
                    """
                    UPDATE tool_operations
                    SET status = ?, error = ?, updated_at = ?
                    WHERE session_key = ? AND status = ?
                      AND expires_at IS NOT NULL AND expires_at <= ?
                    """,
                    (
                        CANCELLED,
                        "确认请求已过期",
                        now,
                        session_key,
                        PENDING_CONFIRMATION,
                        now,
                    ),
                )
                self._conn.commit()
            row = self._conn.execute(
                """
                SELECT * FROM tool_operations
                WHERE session_key = ? AND status = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (session_key, PENDING_CONFIRMATION),
            ).fetchone()
        return self._operation_from_row(row) if row is not None else None

    def transition_operation(
        self,
        operation_id: str,
        *,
        to_status: str,
        allowed_from: Iterable[str] | None = None,
        result: str | None = None,
        error: str | None = None,
    ) -> OperationRecord:
        now = _now()
        with self._lock:
            current = self._conn.execute(
                "SELECT status FROM tool_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if current is None:
                raise KeyError(f"未知操作: {operation_id}")
            allowed = set(allowed_from or ())
            if allowed and str(current["status"]) not in allowed:
                raise ValueError(
                    f"操作 {operation_id} 当前状态 {current['status']}，"
                    f"不能转换为 {to_status}"
                )
            assignments = ["status = ?", "updated_at = ?"]
            params: list[Any] = [to_status, now]
            if result is not None:
                assignments.append("result = ?")
                params.append(result)
            if error is not None:
                assignments.append("error = ?")
                params.append(error)
            params.append(operation_id)
            self._conn.execute(
                f"UPDATE tool_operations SET {', '.join(assignments)} "
                "WHERE operation_id = ?",
                tuple(params),
            )
            self._conn.commit()
        record = self.get_operation(operation_id)
        if record is None:  # pragma: no cover - SQLite invariant
            raise RuntimeError("工具操作状态更新失败")
        return record

    def list_recoverable_operations(self) -> list[OperationRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM tool_operations
                WHERE status IN (?, ?)
                ORDER BY created_at ASC
                """,
                (CONFIRMED, RUNNING),
            ).fetchall()
        return [self._operation_from_row(row) for row in rows]

    def enqueue_outbound(self, payload: dict[str, Any]) -> str:
        outbox_id = uuid.uuid4().hex
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO outbound_outbox (
                    outbox_id, channel, chat_id, payload_json,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'PENDING', ?, ?)
                """,
                (
                    outbox_id,
                    str(payload.get("channel", "")),
                    str(payload.get("chat_id", "")),
                    json.dumps(payload, ensure_ascii=False, default=str),
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return outbox_id

    def list_pending_outbound(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT outbox_id, payload_json, attempts
                FROM outbound_outbox
                WHERE status = 'PENDING'
                ORDER BY created_at ASC LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            payload["outbox_id"] = str(row["outbox_id"])
            payload["attempts"] = int(row["attempts"] or 0)
            results.append(payload)
        return results

    def get_outbound_status(self, outbox_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM outbound_outbox WHERE outbox_id = ?",
                (outbox_id,),
            ).fetchone()
        return str(row["status"]) if row is not None else None

    def stats(self) -> dict[str, int]:
        with self._lock:
            operation_rows = self._conn.execute(
                "SELECT status, COUNT(*) AS count FROM tool_operations GROUP BY status"
            ).fetchall()
            outbox_rows = self._conn.execute(
                "SELECT status, COUNT(*) AS count FROM outbound_outbox GROUP BY status"
            ).fetchall()
        result = {
            f"operations_{str(row['status']).lower()}": int(row["count"])
            for row in operation_rows
        }
        result.update(
            {
                f"outbox_{str(row['status']).lower()}": int(row["count"])
                for row in outbox_rows
            }
        )
        return result

    def get_outbound_status(self, outbox_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM outbound_outbox WHERE outbox_id = ?",
                (outbox_id,),
            ).fetchone()
        return str(row["status"]) if row is not None else None

    def stats(self) -> dict[str, int]:
        with self._lock:
            operation_rows = self._conn.execute(
                "SELECT status, COUNT(*) AS count FROM tool_operations GROUP BY status"
            ).fetchall()
            outbox_rows = self._conn.execute(
                "SELECT status, COUNT(*) AS count FROM outbound_outbox GROUP BY status"
            ).fetchall()
        result = {
            f"operations_{str(row['status']).lower()}": int(row["count"])
            for row in operation_rows
        }
        result.update(
            {
                f"outbox_{str(row['status']).lower()}": int(row["count"])
                for row in outbox_rows
            }
        )
        return result

    def record_outbound_failure(self, outbox_id: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE outbound_outbox
                SET attempts = attempts + 1, last_error = ?, updated_at = ?
                WHERE outbox_id = ? AND status = 'PENDING'
                """,
                (error, _now(), outbox_id),
            )
            self._conn.commit()

    def mark_outbound_sent(self, outbox_id: str) -> None:
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE outbound_outbox
                SET status = 'SENT', updated_at = ?, sent_at = ?
                WHERE outbox_id = ?
                """,
                (now, now, outbox_id),
            )
            self._conn.commit()

    def mark_outbound_dead(self, outbox_id: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE outbound_outbox
                SET status = 'DEAD', last_error = ?, updated_at = ?
                WHERE outbox_id = ? AND status = 'PENDING'
                """,
                (error, _now(), outbox_id),
            )
            self._conn.commit()

    @staticmethod
    def _operation_from_row(row: sqlite3.Row) -> OperationRecord:
        return OperationRecord(
            operation_id=str(row["operation_id"]),
            session_key=str(row["session_key"]),
            channel=str(row["channel"] or ""),
            chat_id=str(row["chat_id"] or ""),
            tool_name=str(row["tool_name"]),
            arguments=json.loads(str(row["arguments_json"])),
            status=str(row["status"]),
            result=str(row["result"] or ""),
            error=str(row["error"] or ""),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            expires_at=str(row["expires_at"]) if row["expires_at"] else None,
        )


__all__ = [
    "CANCELLED",
    "CONFIRMED",
    "FAILED",
    "NEEDS_REVIEW",
    "OperationRecord",
    "PENDING_CONFIRMATION",
    "RUNNING",
    "ReliabilityStore",
    "SUCCEEDED",
]
