"""Qdrant-backed vector index used beside the SQLite document catalog."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from agent.rag.models import (
    ChunkRecord,
    DocumentRecord,
    VectorSearchResult,
)

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    name = "qdrant"

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        api_key: str = "",
        timeout_seconds: float = 10.0,
        connect_attempts: int = 5,
    ) -> None:
        self.url = url.rstrip("/")
        self.collection = collection
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.connect_attempts = max(1, connect_attempts)
        self._client: Any = None
        self._models: Any = None
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    async def initialize(self) -> bool:
        if not self.url:
            logger.warning("Qdrant URL 未配置，向量检索使用 SQLite 基准实现")
            return False
        try:
            from qdrant_client import AsyncQdrantClient, models
        except ImportError:
            logger.warning("未安装 qdrant-client，向量检索使用 SQLite 基准实现")
            return False

        self._models = models
        self._client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key or None,
            timeout=self.timeout_seconds,
        )
        for attempt in range(1, self.connect_attempts + 1):
            try:
                await self._client.get_collections()
                self._available = True
                logger.info("Qdrant 已连接 url=%s collection=%s", self.url, self.collection)
                return True
            except Exception as exc:
                if attempt == self.connect_attempts:
                    logger.warning("Qdrant 连接失败，进入本地降级模式: %s", exc)
                    break
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
        self._available = False
        return False

    async def replace_document(
        self, document: DocumentRecord, chunks: list[ChunkRecord]
    ) -> None:
        self._require_available()
        indexed_chunks = [chunk for chunk in chunks if chunk.embedding]
        if not indexed_chunks:
            raise ValueError("文档没有可写入 Qdrant 的向量")
        vector_size = len(indexed_chunks[0].embedding or ())
        await self._ensure_collection(vector_size)

        existing_ids = await self._document_point_ids(document.id)
        points = [
            self._models.PointStruct(
                id=_point_id(chunk.id),
                vector=list(chunk.embedding or ()),
                payload={
                    "chunk_id": chunk.id,
                    "document_id": document.id,
                    "source_path": document.source_path,
                    "title": document.title,
                    "text": chunk.text,
                    "heading": chunk.heading,
                    "ordinal": chunk.ordinal,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "page_number": chunk.page_number,
                    "content_hash": document.content_hash,
                },
            )
            for chunk in indexed_chunks
        ]
        for start in range(0, len(points), 128):
            await self._client.upsert(
                collection_name=self.collection,
                points=points[start : start + 128],
                wait=True,
            )

        new_ids = {str(point.id) for point in points}
        stale_ids = [point_id for point_id in existing_ids if str(point_id) not in new_ids]
        if stale_ids:
            await self._client.delete(
                collection_name=self.collection,
                points_selector=self._models.PointIdsList(points=stale_ids),
                wait=True,
            )

    async def search(
        self, query_vector: list[float], *, limit: int
    ) -> list[VectorSearchResult]:
        self._require_available()
        response = await self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=max(1, int(limit)),
            with_payload=True,
            with_vectors=False,
        )
        results: list[VectorSearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id", ""))
            if chunk_id:
                results.append(
                    VectorSearchResult(chunk_id=chunk_id, score=float(point.score))
                )
        return results

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
        self._client = None
        self._available = False

    async def _ensure_collection(self, vector_size: int) -> None:
        exists = await self._client.collection_exists(self.collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self.collection,
                vectors_config=self._models.VectorParams(
                    size=vector_size,
                    distance=self._models.Distance.COSINE,
                ),
            )
            return
        info = await self._client.get_collection(self.collection)
        vectors_config = info.config.params.vectors
        configured_size = getattr(vectors_config, "size", None)
        if configured_size is not None and int(configured_size) != vector_size:
            raise ValueError(
                f"Qdrant Collection 向量维度为 {configured_size}，"
                f"当前 Embedding 返回 {vector_size} 维；请新建 Collection 并重新索引"
            )

    async def _document_point_ids(self, document_id: str) -> list[Any]:
        if not await self._client.collection_exists(self.collection):
            return []
        point_ids: list[Any] = []
        offset: Any = None
        query_filter = self._models.Filter(
            must=[
                self._models.FieldCondition(
                    key="document_id",
                    match=self._models.MatchValue(value=document_id),
                )
            ]
        )
        while True:
            records, offset = await self._client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.extend(record.id for record in records)
            if offset is None:
                break
        return point_ids

    def _require_available(self) -> None:
        if not self._available or self._client is None:
            raise RuntimeError("Qdrant 当前不可用")


def _point_id(chunk_id: str) -> str:
    """Qdrant accepts UUID/uint IDs; keep the original chunk id in payload."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"akashic-rag:{chunk_id}"))


__all__ = ["QdrantVectorStore"]
