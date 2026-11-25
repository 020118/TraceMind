from __future__ import annotations

from pathlib import Path

import pytest

from agent.rag.service import RAGService
from agent.rag.models import VectorSearchResult
from agent.rag.store import SQLiteRAGStore
from agent.tools.base import ToolRegistry
from agent.tools.knowledge import register_knowledge_tools


class FakeEmbedder:
    model_name = "fake-embedding"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float("agent" in text.lower()), float("memory" in text.lower()), 1.0]
            for text in texts
        ]

    async def close(self) -> None:
        return None


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return await super().embed(texts)


class FakeVectorStore:
    name = "qdrant"

    def __init__(self) -> None:
        self.available = False
        self.chunks = []

    async def initialize(self) -> bool:
        self.available = True
        return True

    async def replace_document(self, document, chunks) -> None:
        self.chunks = list(chunks)

    async def search(self, query_vector, *, limit: int):
        scored = []
        for chunk in self.chunks:
            vector = chunk.embedding or ()
            score = sum(a * b for a, b in zip(query_vector, vector, strict=True))
            scored.append(VectorSearchResult(chunk_id=chunk.id, score=score))
        return sorted(scored, key=lambda item: -item.score)[:limit]

    async def close(self) -> None:
        self.available = False


@pytest.mark.asyncio
async def test_ingest_search_citation_and_idempotency(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "agent.md"
    source.write_text(
        "# Agent Runtime\n\nAgent loop 负责工具编排。\n\n"
        "# Memory\n\n长期记忆保存用户的稳定偏好。\n",
        encoding="utf-8",
    )
    service = RAGService(
        workspace=tmp_path,
        store=SQLiteRAGStore(tmp_path / "knowledge" / "rag.db"),
        embedder=FakeEmbedder(),
        chunk_size=200,
        chunk_overlap=20,
        top_k=3,
    )
    await service.initialize()

    first = await service.ingest_file("docs/agent.md")
    second = await service.ingest_file("docs/agent.md")
    hits = await service.search("Agent 工具编排")

    assert first.chunk_count == 2
    assert first.semantic_indexed is True
    assert second.unchanged is True
    assert len(service.list_documents()) == 1
    assert hits
    assert hits[0].document.source_path == "docs/agent.md"
    assert "[agent]" in hits[0].citation
    assert "docs/agent.md#L" in hits[0].citation
    await service.close()


@pytest.mark.asyncio
async def test_keyword_fallback_and_workspace_boundary(tmp_path: Path) -> None:
    source = tmp_path / "research.txt"
    source.write_text("RRF 用倒数排名融合关键词检索和向量检索。", encoding="utf-8")
    service = RAGService(
        workspace=tmp_path,
        store=SQLiteRAGStore(tmp_path / "knowledge" / "rag.db"),
        embedder=None,
    )
    await service.initialize()
    result = await service.ingest_file("research.txt")
    hits = await service.search("关键词检索")

    assert result.semantic_indexed is False
    assert hits and "RRF" in hits[0].chunk.text
    with pytest.raises(PermissionError):
        await service.ingest_file(str(tmp_path.parent / "outside.txt"))


@pytest.mark.asyncio
async def test_rag_tools_are_registered_as_deferred_catalog_entries(tmp_path: Path) -> None:
    service = RAGService(
        workspace=tmp_path,
        store=SQLiteRAGStore(tmp_path / "knowledge" / "rag.db"),
    )
    await service.initialize()
    registry = ToolRegistry()
    register_knowledge_tools(registry, service)

    assert registry.get("knowledge_search") is not None
    assert registry.get("knowledge_sync_directory") is not None
    assert registry.get_metadata("knowledge_index_file").risk == "write"
    assert registry.get_always_on_names() == set()
    assert registry.search("文档问答")[0]["name"] == "knowledge_search"


@pytest.mark.asyncio
async def test_qdrant_shadow_metrics_are_recorded(tmp_path: Path) -> None:
    source = tmp_path / "compare.md"
    source.write_text(
        "# Agent\nAgent tool runtime.\n# Memory\nMemory preference store.",
        encoding="utf-8",
    )
    vector_store = FakeVectorStore()
    service = RAGService(
        workspace=tmp_path,
        store=SQLiteRAGStore(tmp_path / "knowledge" / "rag.db"),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        read_from_vector_store=True,
        shadow_evaluation=True,
    )
    await service.initialize()
    ingest = await service.ingest_file("compare.md")
    hits = await service.search("Agent", top_k=2)
    metrics = service.retrieval_metrics()

    assert ingest.vector_store_synced is True
    assert ingest.vector_status == "READY"
    assert hits
    assert metrics["sample_count"] == 1
    assert metrics["qdrant_success_rate"] == 1.0
    assert metrics["average_recall_at_k"] == 1.0
    assert metrics["primary_dense_retriever"] == "qdrant"


@pytest.mark.asyncio
async def test_search_cache_reuses_embedding_and_invalidates_after_reindex(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cache.md"
    source.write_text("# Agent\nAgent runtime and tool calls.", encoding="utf-8")
    embedder = CountingEmbedder()
    service = RAGService(
        workspace=tmp_path,
        store=SQLiteRAGStore(tmp_path / "knowledge" / "rag.db"),
        embedder=embedder,
        search_cache_size=4,
        search_cache_ttl_seconds=60,
    )
    await service.initialize()

    await service.ingest_file("cache.md")
    await service.search("agent runtime")
    await service.search("  agent   runtime  ")
    assert embedder.calls == 2  # one ingestion call + one query embedding call
    assert service.retrieval_metrics()["search_cache"]["hits"] == 1

    source.write_text("# Memory\nMemory store and stable preferences.", encoding="utf-8")
    await service.ingest_file("cache.md")
    hits = await service.search("memory store")
    assert hits and "Memory" in hits[0].chunk.text
    assert embedder.calls == 4  # re-indexing and the post-update query were uncached
    assert service.retrieval_metrics()["search_cache"]["entries"] == 1
    await service.close()


@pytest.mark.asyncio
async def test_directory_sync_is_incremental_and_reports_unsupported_files(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.md").write_text("# Agent\nAgent runtime.", encoding="utf-8")
    (docs / "two.txt").write_text("Memory store.", encoding="utf-8")
    (docs / "ignored.bin").write_bytes(b"not a document")
    service = RAGService(
        workspace=tmp_path,
        store=SQLiteRAGStore(tmp_path / "knowledge" / "rag.db"),
        search_cache_size=0,
    )
    await service.initialize()

    first = await service.ingest_directory("docs")
    second = await service.ingest_directory("docs")

    assert first["selected_files"] == 2
    assert first["indexed"] == 2
    assert first["unsupported_files"] == 1
    assert first["skipped_files"] == 0
    assert second["indexed"] == 0
    assert second["unchanged"] == 2
    assert len(service.list_documents()) == 2
    await service.close()


@pytest.mark.asyncio
async def test_mmr_diversification_prevents_one_document_from_filling_top_k(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    repeated = "\n\n".join(
        f"## Part {index}\nAgent runtime design and tool orchestration. "
        + ("Agent details. " * 20)
        for index in range(4)
    )
    (docs / "a.md").write_text(repeated, encoding="utf-8")
    (docs / "b.md").write_text(
        "# Agent\nAgent memory integration from another source.",
        encoding="utf-8",
    )
    service = RAGService(
        workspace=tmp_path,
        store=SQLiteRAGStore(tmp_path / "knowledge" / "rag.db"),
        embedder=FakeEmbedder(),
        chunk_size=200,
        chunk_overlap=20,
        search_cache_size=0,
    )
    await service.initialize()
    await service.ingest_directory("docs")

    raw_hits = await service.search(
        "Agent", top_k=3, diversify=False, max_per_document=0
    )
    diverse_hits = await service.search(
        "Agent", top_k=3, diversify=True, max_per_document=1
    )
    single_document_hits = await service.search(
        "runtime", top_k=3, diversify=True, max_per_document=1
    )

    assert len({hit.document.source_path for hit in raw_hits}) == 1
    assert len({hit.document.source_path for hit in diverse_hits}) == 2
    assert len(single_document_hits) == 3
    await service.close()

