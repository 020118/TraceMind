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

