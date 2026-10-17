"""SQLite implementation of the replaceable RAG store port."""

from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path

from agent.rag.models import ChunkRecord, DocumentRecord


class SQLiteRAGStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    vector_status TEXT NOT NULL DEFAULT 'PENDING',
                    vector_error TEXT NOT NULL DEFAULT '',
                    vector_indexed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    heading TEXT NOT NULL DEFAULT '',
                    start_line INTEGER,
                    end_line INTEGER,
                    page_number INTEGER,
                    embedding_json TEXT,
                    UNIQUE(document_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_document
                    ON rag_chunks(document_id, ordinal);
                CREATE TABLE IF NOT EXISTS rag_retrieval_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    exact_latency_ms REAL,
                    qdrant_latency_ms REAL,
                    recall_at_k REAL,
                    qdrant_success INTEGER NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._ensure_document_columns(conn)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM rag_documents WHERE id = ?", (document_id,)
            ).fetchone()
        return _document_from_row(row) if row else None

    def replace_document(
        self, document: DocumentRecord, chunks: list[ChunkRecord]
    ) -> None:
        """Atomically replace one document and all of its chunks."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO rag_documents
                    (id, source_path, title, media_type, content_hash, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_path=excluded.source_path,
                    title=excluded.title,
                    media_type=excluded.media_type,
                    content_hash=excluded.content_hash,
                    chunk_count=excluded.chunk_count,
                    vector_status='INDEXING',
                    vector_error='',
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    document.id, document.source_path, document.title,
                    document.media_type, document.content_hash, document.chunk_count,
                ),
            )
            conn.execute("DELETE FROM rag_chunks WHERE document_id = ?", (document.id,))
            conn.executemany(
                """
                INSERT INTO rag_chunks
                    (id, document_id, ordinal, text, heading, start_line,
                     end_line, page_number, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id, chunk.document_id, chunk.ordinal, chunk.text,
                        chunk.heading, chunk.start_line, chunk.end_line,
                        chunk.page_number,
                        json.dumps(chunk.embedding) if chunk.embedding is not None else None,
                    )
                    for chunk in chunks
                ],
            )

    @staticmethod
    def _ensure_document_columns(conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(rag_documents)").fetchall()
        }
        additions = {
            "vector_status": "TEXT NOT NULL DEFAULT 'PENDING'",
            "vector_error": "TEXT NOT NULL DEFAULT ''",
            "vector_indexed_at": "TEXT",
        }
        for name, definition in additions.items():
            if name not in columns:
                conn.execute(
                    f"ALTER TABLE rag_documents ADD COLUMN {name} {definition}"
                )

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM rag_documents ORDER BY updated_at DESC, title"
            ).fetchall()
        return [_document_from_row(row) for row in rows]

    def list_chunks(self) -> list[tuple[ChunkRecord, DocumentRecord]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*, d.source_path, d.title, d.media_type, d.content_hash,
                       d.chunk_count, d.created_at, d.updated_at,
                       d.vector_status, d.vector_error, d.vector_indexed_at
                FROM rag_chunks c
                JOIN rag_documents d ON d.id = c.document_id
                ORDER BY d.title, c.ordinal
                """
            ).fetchall()
        results: list[tuple[ChunkRecord, DocumentRecord]] = []
        for row in rows:
            embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else None
            chunk = ChunkRecord(
                id=row["id"], document_id=row["document_id"], ordinal=row["ordinal"],
                text=row["text"], heading=row["heading"], start_line=row["start_line"],
                end_line=row["end_line"], page_number=row["page_number"],
                embedding=tuple(float(value) for value in embedding) if embedding else None,
            )
            document = DocumentRecord(
                id=row["document_id"], source_path=row["source_path"], title=row["title"],
                media_type=row["media_type"], content_hash=row["content_hash"],
                chunk_count=row["chunk_count"], created_at=row["created_at"],
                updated_at=row["updated_at"],
                vector_status=row["vector_status"], vector_error=row["vector_error"],
                vector_indexed_at=row["vector_indexed_at"] or "",
            )
            results.append((chunk, document))
        return results

    def get_chunks_for_document(self, document_id: str) -> list[ChunkRecord]:
        return [
            chunk
            for chunk, document in self.list_chunks()
            if document.id == document_id
        ]

    def list_documents_for_vector_sync(self) -> list[DocumentRecord]:
        return [
            document
            for document in self.list_documents()
            if document.vector_status != "READY"
        ]

    def update_vector_status(
        self, document_id: str, status: str, error: str = ""
    ) -> None:
        indexed_at = "CURRENT_TIMESTAMP" if status == "READY" else "NULL"
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE rag_documents
                SET vector_status=?, vector_error=?, vector_indexed_at={indexed_at},
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (status, error[:2000], document_id),
            )

    def record_retrieval_metric(
        self,
        *,
        query_hash: str,
        top_k: int,
        exact_latency_ms: float | None,
        qdrant_latency_ms: float | None,
        recall_at_k: float | None,
        qdrant_success: bool,
        error: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rag_retrieval_metrics
                    (query_hash, top_k, exact_latency_ms, qdrant_latency_ms,
                     recall_at_k, qdrant_success, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_hash, top_k, exact_latency_ms, qdrant_latency_ms,
                    recall_at_k, int(qdrant_success), error[:2000],
                ),
            )

    def retrieval_metrics_summary(self, limit: int = 500) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM rag_retrieval_metrics
                ORDER BY id DESC LIMIT ?
                """,
                (max(1, min(int(limit), 5000)),),
            ).fetchall()
        exact = [float(row["exact_latency_ms"]) for row in rows if row["exact_latency_ms"] is not None]
        qdrant = [float(row["qdrant_latency_ms"]) for row in rows if row["qdrant_latency_ms"] is not None]
        recalls = [float(row["recall_at_k"]) for row in rows if row["recall_at_k"] is not None]
        successes = sum(int(row["qdrant_success"]) for row in rows)
        return {
            "sample_count": len(rows),
            "qdrant_success_count": successes,
            "qdrant_error_count": len(rows) - successes,
            "qdrant_success_rate": round(successes / len(rows), 4) if rows else None,
            "average_recall_at_k": round(statistics.fmean(recalls), 4) if recalls else None,
            "exact_latency_ms": _latency_summary(exact),
            "qdrant_latency_ms": _latency_summary(qdrant),
            "average_speedup": round(statistics.fmean(exact) / statistics.fmean(qdrant), 3)
            if exact and qdrant and statistics.fmean(qdrant) > 0 else None,
        }

    def close(self) -> None:
        return None


def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"], source_path=row["source_path"], title=row["title"],
        media_type=row["media_type"], content_hash=row["content_hash"],
        chunk_count=row["chunk_count"], created_at=row["created_at"],
        updated_at=row["updated_at"],
        vector_status=row["vector_status"], vector_error=row["vector_error"],
        vector_indexed_at=row["vector_indexed_at"] or "",
    )


def _latency_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "average": None, "p50": None, "p95": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "average": round(statistics.fmean(values), 3),
        "p50": round(_percentile(ordered, 0.50), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


__all__ = ["SQLiteRAGStore"]
