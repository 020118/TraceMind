"""One-shot smoke test for the project's QdrantVectorStore adapter."""

from __future__ import annotations

import asyncio
import os
import uuid

from qdrant_client import AsyncQdrantClient

from agent.rag.models import ChunkRecord, DocumentRecord
from agent.rag.qdrant_store import QdrantVectorStore


async def main() -> None:
    url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    collection = f"agent_rag_smoke_{uuid.uuid4().hex}"
    store = QdrantVectorStore(url=url, collection=collection)
    cleanup = AsyncQdrantClient(url=url)
    document = DocumentRecord(
        id="smoke-doc", source_path="smoke.md", title="smoke",
        media_type="text/markdown", content_hash="v1", chunk_count=2,
    )
    chunks = [
        ChunkRecord(
            id="smoke-agent", document_id=document.id, ordinal=0,
            text="agent runtime", embedding=(1.0, 0.0, 0.0),
        ),
        ChunkRecord(
            id="smoke-memory", document_id=document.id, ordinal=1,
            text="memory store", embedding=(0.0, 1.0, 0.0),
        ),
    ]
    try:
        if not await store.initialize():
            raise RuntimeError("Qdrant unavailable")
        await store.replace_document(document, chunks)
        first = await store.search([1.0, 0.0, 0.0], limit=2)
        if not first or first[0].chunk_id != "smoke-agent":
            raise AssertionError(f"unexpected ranking: {first}")
        await store.replace_document(document, chunks[:1])
        second = await store.search([0.0, 1.0, 0.0], limit=2)
        if {item.chunk_id for item in second} != {"smoke-agent"}:
            raise AssertionError(f"stale point was not removed: {second}")
        print("QdrantVectorStore smoke test passed")
    finally:
        await store.close()
        if await cleanup.collection_exists(collection):
            await cleanup.delete_collection(collection)
        await cleanup.close()


if __name__ == "__main__":
    asyncio.run(main())
