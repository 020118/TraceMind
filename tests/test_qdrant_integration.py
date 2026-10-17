from __future__ import annotations

import os
import uuid

import pytest

from agent.rag.models import ChunkRecord, DocumentRecord
from agent.rag.qdrant_store import QdrantVectorStore


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("QDRANT_INTEGRATION_URL"),
    reason="set QDRANT_INTEGRATION_URL to run the real Qdrant integration test",
)
async def test_qdrant_replace_search_and_remove_stale_points() -> None:
    from qdrant_client import AsyncQdrantClient

    url = os.environ["QDRANT_INTEGRATION_URL"]
    collection = f"agent_rag_test_{uuid.uuid4().hex}"
    store = QdrantVectorStore(url=url, collection=collection)
    cleanup = AsyncQdrantClient(url=url)
    document = DocumentRecord(
        id="doc-1",
        source_path="knowledge_docs/test.md",
        title="test",
        media_type="text/markdown",
        content_hash="v1",
        chunk_count=2,
    )
    chunks = [
        ChunkRecord(
            id="chunk-a", document_id="doc-1", ordinal=0,
            text="agent runtime", embedding=(1.0, 0.0, 0.0),
        ),
        ChunkRecord(
            id="chunk-b", document_id="doc-1", ordinal=1,
            text="memory store", embedding=(0.0, 1.0, 0.0),
        ),
    ]
    try:
        assert await store.initialize() is True
        await store.replace_document(document, chunks)
        first = await store.search([1.0, 0.0, 0.0], limit=2)
        assert first[0].chunk_id == "chunk-a"

        await store.replace_document(document, chunks[:1])
        second = await store.search([0.0, 1.0, 0.0], limit=2)
        assert {item.chunk_id for item in second} == {"chunk-a"}
    finally:
        await store.close()
        if await cleanup.collection_exists(collection):
            await cleanup.delete_collection(collection)
        await cleanup.close()
