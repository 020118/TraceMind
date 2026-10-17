"""RAG ingestion and hybrid retrieval service."""

from __future__ import annotations

import hashlib
import logging
import math
import mimetypes
import re
import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

from agent.rag.chunking import chunk_sections, parse_document
from agent.rag.models import (
    ChunkRecord,
    DocumentRecord,
    EmbeddingProvider,
    IngestResult,
    RankedCandidate,
    RAGStore,
    SearchHit,
    VectorStore,
)
from agent.rag.qdrant_store import QdrantVectorStore
from agent.rag.store import SQLiteRAGStore

logger = logging.getLogger(__name__)
_LATIN_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


class OpenAIEmbeddingProvider:
    """Embedding adapter for OpenAI-compatible APIs (including DashScope)."""

    def __init__(self, *, api_key: str, model: str, base_url: str = "") -> None:
        options: dict[str, str] = {"api_key": api_key}
        if base_url.strip():
            options["base_url"] = base_url.strip()
        self._client = AsyncOpenAI(**options)
        self.model_name = model.strip()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            response = await self._client.embeddings.create(
                model=self.model_name,
                input=texts[start : start + 64],
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(
                [[float(value) for value in item.embedding] for item in ordered]
            )
        return vectors

    async def close(self) -> None:
        await self._client.close()


class RAGService:
    """Indexes workspace documents and retrieves cited evidence.

    SQLite currently performs exact vector scanning, which is appropriate for a
    personal knowledge base. The service depends on ``RAGStore`` and
    ``EmbeddingProvider`` ports so pgvector/Qdrant can replace them later without
    changing the Agent tools.
    """

    def __init__(
        self,
        *,
        workspace: Path,
        store: RAGStore,
        embedder: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        read_from_vector_store: bool = True,
        shadow_evaluation: bool = True,
        sync_on_start: bool = True,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        top_k: int = 6,
        rrf_k: int = 60,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.store = store
        self.embedder = embedder
        self.vector_store = vector_store
        self.read_from_vector_store = read_from_vector_store
        self.shadow_evaluation = shadow_evaluation
        self.sync_on_start = sync_on_start
        self.chunk_size = max(200, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size // 2))
        self.top_k = max(1, min(int(top_k), 20))
        self.rrf_k = max(1, int(rrf_k))

    async def initialize(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.store.initialize()
        if self.vector_store is not None and await self.vector_store.initialize():
            if self.sync_on_start:
                await self._sync_pending_documents()

    async def ingest_file(self, path: str, *, title: str = "") -> IngestResult:
        source = self._resolve_workspace_path(path)
        if not source.is_file():
            raise FileNotFoundError(f"知识库文件不存在: {path}")
        if source.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("单个知识库文件不能超过 20 MB")

        relative_path = source.relative_to(self.workspace).as_posix()
        raw_bytes = source.read_bytes()
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        document_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
        existing = self.store.get_document(document_id)
        if existing is not None and existing.content_hash == content_hash:
            vector_synced = existing.vector_status == "READY"
            if (
                not vector_synced
                and self.vector_store is not None
                and self.vector_store.available
            ):
                chunks = self.store.get_chunks_for_document(document_id)
                vector_synced = await self._sync_vector_store(existing, chunks)
                existing = self.store.get_document(document_id) or existing
            return IngestResult(
                document_id=document_id,
                source_path=relative_path,
                chunk_count=existing.chunk_count,
                unchanged=True,
                semantic_indexed=self.embedder is not None,
                vector_store_synced=vector_synced,
                vector_status=existing.vector_status,
            )

        sections = parse_document(source)
        parsed_chunks = chunk_sections(
            sections,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not parsed_chunks:
            raise ValueError("文件中没有可建立索引的文本")

        embeddings: list[list[float]] | None = None
        if self.embedder is not None:
            try:
                search_texts = [
                    f"{chunk.heading}\n{chunk.text}".strip() for chunk in parsed_chunks
                ]
                embeddings = await self.embedder.embed(search_texts)
                if len(embeddings) != len(parsed_chunks):
                    raise ValueError("embedding 返回数量与文档分块数量不一致")
            except Exception as exc:
                logger.warning("RAG 向量索引失败，保留关键词索引: %s", exc)
                embeddings = None

        chunks = [
            ChunkRecord(
                id=hashlib.sha256(
                    f"{document_id}:{index}:{content_hash}".encode("utf-8")
                ).hexdigest()[:32],
                document_id=document_id,
                ordinal=index,
                text=chunk.text,
                heading=chunk.heading,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                page_number=chunk.page_number,
                embedding=tuple(embeddings[index]) if embeddings is not None else None,
            )
            for index, chunk in enumerate(parsed_chunks)
        ]
        document = DocumentRecord(
            id=document_id,
            source_path=relative_path,
            title=title.strip() or source.stem,
            media_type=mimetypes.guess_type(source.name)[0] or "text/plain",
            content_hash=content_hash,
            chunk_count=len(chunks),
        )
        self.store.replace_document(document, chunks)
        vector_synced = await self._sync_vector_store(document, chunks)
        if self.vector_store is None:
            self.store.update_vector_status(document.id, "LOCAL_ONLY")
        stored_document = self.store.get_document(document.id) or document
        return IngestResult(
            document_id=document_id,
            source_path=relative_path,
            chunk_count=len(chunks),
            semantic_indexed=embeddings is not None,
            vector_store_synced=vector_synced,
            vector_status=stored_document.vector_status,
        )

    async def search(self, query: str, *, top_k: int | None = None) -> list[SearchHit]:
        query = (query or "").strip()
        if not query:
            return []
        records = self.store.list_chunks()
        if not records:
            return []
        limit = max(1, min(int(top_k or self.top_k), 20))
        candidate_limit = min(len(records), max(limit * 4, 20))
        candidates = {
            chunk.id: RankedCandidate(chunk=chunk, document=document)
            for chunk, document in records
        }

        keyword_ranking: list[RankedCandidate] = []
        for candidate in candidates.values():
            candidate.keyword_score = _keyword_similarity(
                query, f"{candidate.chunk.heading} {candidate.chunk.text}"
            )
            if candidate.keyword_score > 0:
                keyword_ranking.append(candidate)
        keyword_ranking.sort(
            key=lambda item: (-item.keyword_score, item.document.title, item.chunk.ordinal)
        )
        keyword_ranking = keyword_ranking[:candidate_limit]

        exact_ranking: list[RankedCandidate] = []
        qdrant_ranking: list[RankedCandidate] = []
        exact_latency_ms: float | None = None
        qdrant_latency_ms: float | None = None
        qdrant_success = False
        qdrant_error = ""
        if self.embedder is not None and any(
            candidate.chunk.embedding for candidate in candidates.values()
        ):
            try:
                query_vectors = await self.embedder.embed([query])
                if query_vectors:
                    query_vector = query_vectors[0]
                    exact_started = time.perf_counter()
                    for candidate in candidates.values():
                        if candidate.chunk.embedding:
                            exact_score = _cosine_similarity(
                                query_vector, candidate.chunk.embedding
                            )
                            if exact_score > 0:
                                exact_ranking.append(
                                    RankedCandidate(
                                        chunk=candidate.chunk,
                                        document=candidate.document,
                                        semantic_score=exact_score,
                                        keyword_score=candidate.keyword_score,
                                    )
                                )
                    exact_ranking.sort(
                        key=lambda item: (
                            -item.semantic_score,
                            item.document.title,
                            item.chunk.ordinal,
                        )
                    )
                    exact_ranking = exact_ranking[:candidate_limit]
                    exact_latency_ms = (time.perf_counter() - exact_started) * 1000

                    if self.vector_store is not None and self.vector_store.available:
                        qdrant_started = time.perf_counter()
                        try:
                            vector_hits = await self.vector_store.search(
                                query_vector, limit=candidate_limit
                            )
                            qdrant_latency_ms = (
                                time.perf_counter() - qdrant_started
                            ) * 1000
                            for hit in vector_hits:
                                candidate = candidates.get(hit.chunk_id)
                                if candidate is not None:
                                    qdrant_ranking.append(
                                        RankedCandidate(
                                            chunk=candidate.chunk,
                                            document=candidate.document,
                                            semantic_score=hit.score,
                                            keyword_score=candidate.keyword_score,
                                        )
                                    )
                            qdrant_success = True
                        except Exception as exc:
                            qdrant_latency_ms = (
                                time.perf_counter() - qdrant_started
                            ) * 1000
                            qdrant_error = str(exc)
                            logger.warning(
                                "Qdrant 检索失败，回退到 SQLite 精确检索: %s", exc
                            )
            except Exception as exc:
                logger.warning("RAG 语义检索失败，降级为关键词检索: %s", exc)

        semantic_ranking = (
            qdrant_ranking
            if self.read_from_vector_store and qdrant_success and qdrant_ranking
            else exact_ranking
        )
        if self.vector_store is not None and (self.shadow_evaluation or qdrant_error):
            recall = _recall_at_k(exact_ranking, qdrant_ranking, limit)
            self.store.record_retrieval_metric(
                query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
                top_k=limit,
                exact_latency_ms=exact_latency_ms,
                qdrant_latency_ms=qdrant_latency_ms,
                recall_at_k=recall if qdrant_success else None,
                qdrant_success=qdrant_success,
                error=qdrant_error,
            )

        fused = reciprocal_rank_fusion(
            {"keyword": keyword_ranking, "semantic": semantic_ranking},
            k=self.rrf_k,
        )
        return [
            SearchHit(
                chunk=candidate.chunk,
                document=candidate.document,
                score=round(score, 8),
                semantic_score=round(candidate.semantic_score, 6),
                keyword_score=round(candidate.keyword_score, 6),
            )
            for candidate, score in fused[:limit]
        ]

    def list_documents(self) -> list[DocumentRecord]:
        return self.store.list_documents()

    def retrieval_metrics(self, limit: int = 500) -> dict:
        summary = self.store.retrieval_metrics_summary(limit)
        summary.update(
            {
                "primary_dense_retriever": (
                    self.vector_store.name
                    if self.vector_store is not None
                    and self.vector_store.available
                    and self.read_from_vector_store
                    else "sqlite_exact"
                ),
                "vector_store_available": bool(
                    self.vector_store is not None and self.vector_store.available
                ),
                "metric_definition": {
                    "recall_at_k": "Qdrant Top-K 与 SQLite 精确 Dense Top-K 的重合比例",
                    "speedup": "SQLite 精确扫描平均延迟 / Qdrant 平均延迟",
                },
            }
        )
        return summary

    async def close(self) -> None:
        if self.vector_store is not None:
            await self.vector_store.close()
        if self.embedder is not None:
            await self.embedder.close()
        self.store.close()

    async def _sync_pending_documents(self) -> None:
        for document in self.store.list_documents_for_vector_sync():
            chunks = self.store.get_chunks_for_document(document.id)
            if any(chunk.embedding for chunk in chunks):
                await self._sync_vector_store(document, chunks)

    async def _sync_vector_store(
        self, document: DocumentRecord, chunks: list[ChunkRecord]
    ) -> bool:
        if not any(chunk.embedding for chunk in chunks):
            self.store.update_vector_status(document.id, "NO_EMBEDDING")
            return False
        if self.vector_store is None:
            return False
        if not self.vector_store.available:
            self.store.update_vector_status(document.id, "FAILED", "向量数据库不可用")
            return False
        self.store.update_vector_status(document.id, "INDEXING")
        try:
            await self.vector_store.replace_document(document, chunks)
            self.store.update_vector_status(document.id, "READY")
            return True
        except Exception as exc:
            logger.warning("文档同步到 %s 失败: %s", self.vector_store.name, exc)
            self.store.update_vector_status(document.id, "FAILED", str(exc))
            return False

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.workspace):
            raise PermissionError("知识库只能读取 workspace 内的文件")
        return resolved


def reciprocal_rank_fusion(
    rankings: dict[str, list[RankedCandidate]], *, k: int = 60
) -> list[tuple[RankedCandidate, float]]:
    """Fuse independent rankings without comparing incompatible raw scores."""
    scores: dict[str, float] = {}
    candidates: dict[str, RankedCandidate] = {}
    for ranking_name, ranking in rankings.items():
        for rank, candidate in enumerate(ranking, start=1):
            candidate.ranks[ranking_name] = rank
            candidates[candidate.chunk.id] = candidate
            scores[candidate.chunk.id] = scores.get(candidate.chunk.id, 0.0) + 1.0 / (k + rank)
    fused = [(candidates[chunk_id], score) for chunk_id, score in scores.items()]
    fused.sort(
        key=lambda item: (
            -item[1], -item[0].keyword_score, -item[0].semantic_score,
            item[0].document.title, item[0].chunk.ordinal,
        )
    )
    return fused


def create_rag_service(
    *,
    workspace: Path,
    embedding_config: dict[str, str] | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    top_k: int = 6,
    rrf_k: int = 60,
    vector_store_config: dict[str, object] | None = None,
) -> RAGService:
    embedding_config = embedding_config or {}
    api_key = embedding_config.get("api_key", "").strip()
    model = embedding_config.get("model", "").strip()
    embedder: EmbeddingProvider | None = None
    if api_key and model:
        embedder = OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model,
            base_url=embedding_config.get("base_url", ""),
        )
    vector_store_config = vector_store_config or {}
    vector_store: VectorStore | None = None
    if str(vector_store_config.get("provider", "sqlite")).lower() == "qdrant":
        vector_store = QdrantVectorStore(
            url=str(vector_store_config.get("url", "")),
            api_key=str(vector_store_config.get("api_key", "")),
            collection=str(
                vector_store_config.get("collection", "agent_knowledge_v1")
            ),
            timeout_seconds=float(vector_store_config.get("timeout_seconds", 10.0)),
        )
    service = RAGService(
        workspace=workspace,
        store=SQLiteRAGStore(Path(workspace) / "knowledge" / "rag.db"),
        embedder=embedder,
        vector_store=vector_store,
        read_from_vector_store=bool(
            vector_store_config.get("read_from_qdrant", True)
        ),
        shadow_evaluation=bool(
            vector_store_config.get("shadow_evaluation", True)
        ),
        sync_on_start=bool(vector_store_config.get("sync_on_start", True)),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        rrf_k=rrf_k,
    )
    return service


def _recall_at_k(
    exact: list[RankedCandidate], approximate: list[RankedCandidate], k: int
) -> float | None:
    exact_ids = [item.chunk.id for item in exact[:k]]
    if not exact_ids:
        return None
    approximate_ids = {item.chunk.id for item in approximate[:k]}
    return round(len(set(exact_ids) & approximate_ids) / len(exact_ids), 6)


def _tokens(text: str) -> set[str]:
    lowered = (text or "").lower()
    tokens = {match.group(0) for match in _LATIN_RE.finditer(lowered)}
    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _keyword_similarity(query: str, candidate: str) -> float:
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    if not overlap:
        return 0.0
    score = overlap / math.sqrt(len(query_tokens) * len(candidate_tokens))
    normalized_query = re.sub(r"\s+", "", query.lower())
    normalized_candidate = re.sub(r"\s+", "", candidate.lower())
    if normalized_query in normalized_candidate:
        score += 0.2
    return min(score, 1.0)


def _cosine_similarity(left: list[float], right: tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_vector = np.asarray(left, dtype=np.float32)
    right_vector = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator <= 0:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)


__all__ = [
    "OpenAIEmbeddingProvider",
    "RAGService",
    "create_rag_service",
    "reciprocal_rank_fusion",
]

