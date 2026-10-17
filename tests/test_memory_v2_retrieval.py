from __future__ import annotations

import pytest

from agent.memory.retrieval2 import HybridMemoryRetriever
from agent.memory.store2 import MemoryStoreConfig, MemoryStoreV2


@pytest.fixture
async def v2_store(tmp_path):
    store = MemoryStoreV2(MemoryStoreConfig(workspace=tmp_path))
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_keyword_retrieval_only_returns_relevant_chunks(v2_store, tmp_path):
    v2_store.write_long_term(
        "# 运动偏好\n- 用户喜欢跑步\n\n# 饮品偏好\n- 用户喜欢黑咖啡"
    )
    v2_store.append_pending("用户正在学习 FastAPI", "turn-1")
    v2_store.write_recent_context("最近在讨论 Docker 部署。")
    retriever = HybridMemoryRetriever(v2_store, workspace=tmp_path, top_k=2)

    context = await retriever.build_context("请推荐一些跑步路线")

    assert "用户喜欢跑步" in context
    assert "黑咖啡" not in context
    assert "FastAPI" not in context


@pytest.mark.asyncio
async def test_semantic_retrieval_ranks_meaning_and_reuses_cache(v2_store, tmp_path):
    v2_store.write_long_term(
        "# 研究方向\n- 用户关注个性化论文推荐系统\n- 用户周末喜欢制作手冲咖啡"
    )
    calls: list[list[str]] = []

    async def fake_embedder(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        vectors = []
        for text in texts:
            if "研究资料" in text or "论文推荐" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors

    retriever = HybridMemoryRetriever(
        v2_store,
        workspace=tmp_path,
        model="fake-embedding",
        top_k=1,
        score_threshold=0.6,
        semantic_weight=1.0,
        keyword_weight=0.0,
        embedder=fake_embedder,
    )

    first = await retriever.search("帮我找新的研究资料")
    second = await retriever.search("帮我找新的研究资料")

    assert first[0].text == "用户关注个性化论文推荐系统"
    assert second == first
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_embedding_failure_falls_back_to_keyword(v2_store, tmp_path):
    v2_store.write_long_term("# 偏好\n- 用户喜欢在松山湖跑步")

    async def unavailable_embedder(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service unavailable")

    retriever = HybridMemoryRetriever(
        v2_store,
        workspace=tmp_path,
        model="fake-embedding",
        embedder=unavailable_embedder,
    )

    hits = await retriever.search("松山湖跑步路线")

    assert hits
    assert "松山湖跑步" in hits[0].text
