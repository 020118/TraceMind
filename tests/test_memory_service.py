""" MemoryService 测试

运行方式：
    pytest tests/test_memory_service.py -v

关键技巧：
    用 pytest 的 monkeypatch 替换 service.embed 方法，
    避免测试时真的调用 embedding API（网络请求、花钱、慢）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from agent.memory.service import FACT_EXTRACTION_PROMPT, MemoryService
from agent.memory.store import MemoryStore


# 一个固定的假向量，用于替换真实 embedding
FAKE_EMB = [0.1] * 128


@pytest.fixture
async def service(monkeypatch):
    """创建使用内存数据库的 MemoryService，并 mock 掉 embed"""
    store = MemoryStore(":memory:")
    await store.initialize()

    # 构造 service 需要 api_key，但因为我们 mock 了 embed，不会真的调用
    svc = MemoryService(
        store=store,
        embed_api_key="test-key",
        embed_base_url="https://example.com/v1",
        embed_model="test-model",
    )

    # ── 关键：用 monkeypatch 替换 embed ──
    # 用文本作为随机种子生成确定性向量：
    #   相同文本 → 相同向量（去重测试需要）
    #   不同文本 → 不同向量（区分不同记忆）
    import random
    async def fake_embed(text):
        rng = random.Random(text)  # 文本作种子，可复现
        return [rng.random() for _ in range(128)]

    monkeypatch.setattr(svc, "embed", fake_embed)

    yield svc
    await store.close()


# ── add_memory 去重测试 ────────────────────────────────

async def test_add_memory_basic(service):
    """添加一条记忆，能在 store 里查到"""
    mem_id = await service.add_memory("用户喜欢跑步", memory_type="preference")
    item = await service.store.get(mem_id)
    assert item is not None
    assert item.content == "用户喜欢跑步"
    assert item.memory_type == "preference"


async def test_add_memory_dedup(service):
    """相同内容去重：第二次写入返回同一个 id"""
    first_id = await service.add_memory("用户喜欢跑步", memory_type="preference")
    second_id = await service.add_memory("用户喜欢跑步", memory_type="preference")
    assert first_id == second_id


# ── 去重边界 ────────────────────────────────────────────

async def test_add_memory_no_dedup_different(service):
    """不同内容不去重：两个 id 不同"""
    id1 = await service.add_memory("用户喜欢跑步", memory_type="preference")
    id2 = await service.add_memory("用户喜欢打羽毛球", memory_type="preference")
    assert id1 != id2


# ── search 测试 ─────────────────────────────────────────

async def test_search_returns_matching(service):
    """搜索能返回匹配的记忆"""
    await service.add_memory("用户喜欢喝咖啡", memory_type="preference")
    results = await service.search("用户喜欢喝咖啡")
    assert len(results) >= 1


# ── build_memory_context 测试 ──────────────────────────

async def test_build_context_returns_string(service):
    """构建上下文块"""
    await service.add_memory("用户喜欢跑步", memory_type="preference")
    context = await service.build_memory_context("用户喜欢跑步")
    assert isinstance(context, str)
    assert "用户喜欢跑步" in context


async def test_build_context_empty(service):
    """没有记忆时返回空字符串"""
    context = await service.build_memory_context("不存在的记忆")
    assert context == ""


# ── 事实萃取测试（需要 mock fast_llm）──────────────────

class FakeLLMResponse:
    """假的 LLM 响应 —— extract_facts 只用到 .content 一个属性"""
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """假的 fast_llm —— 返回预设内容，并记录每次 chat 的入参

    这样测试就能断言：
        1. extract_facts 把什么 messages 传给了 LLM（验证只含用户消息）
        2. LLM 返回各种内容时，extract_facts 的解析表现
    """
    def __init__(self, content: str = "[]"):
        self.content = content
        self.calls: list[dict] = []   # 每次 chat 调用的入参快照

    async def chat(self, messages, tools=None, tool_choice="auto"):
        self.calls.append({"messages": messages, "tools": tools})
        return FakeLLMResponse(self.content)


@pytest.fixture
async def llm_service(monkeypatch):
    """带假 fast_llm 的 service —— 专门测 extract_facts"""
    store = MemoryStore(":memory:")
    await store.initialize()

    fake_llm = FakeLLM()
    svc = MemoryService(
        store=store,
        embed_api_key="test-key",
        embed_base_url="https://example.com/v1",
        embed_model="test-model",
        fast_llm=fake_llm,  # type: ignore — 测试用鸭子类型，不需要真的 LLMClient
    )

    # 和上面的 service fixture 一样，mock 掉 embed，避免真调用 API
    import random
    async def fake_embed(text):
        rng = random.Random(text)
        return [rng.random() for _ in range(128)]

    monkeypatch.setattr(svc, "embed", fake_embed)

    yield svc, fake_llm
    await store.close()


async def test_extract_no_fast_llm_returns_empty(service):
    """没配置 fast_llm 时，萃取直接返回空，不报错"""
    facts = await service.extract_facts("我叫张三，住在东莞", "给你推荐了几个地方")
    assert facts == []


async def test_extract_empty_message_returns_empty(llm_service):
    """空白用户消息不调用 LLM，直接返回空"""
    svc, fake_llm = llm_service
    facts = await svc.extract_facts("   ", "回复")
    assert facts == []
    assert fake_llm.calls == []   # 关键：LLM 根本没被调用


async def test_extract_returns_parsed_facts(llm_service):
    """正常情况：LLM 返回 JSON 数组 → 解析成事实列表"""
    svc, fake_llm = llm_service
    fake_llm.content = '["用户叫张三", "用户住在东莞"]'

    facts = await svc.extract_facts("我叫张三，住在东莞", "给你推荐了几个地方")

    assert facts == ["用户叫张三", "用户住在东莞"]


async def test_extract_handles_json_codeblock(llm_service):
    """LLM 返回用 markdown 代码块包着的 JSON → 也能解析"""
    svc, fake_llm = llm_service
    fake_llm.content = '```json\n["用户喜欢跑步"]\n```'

    facts = await svc.extract_facts("我喜欢跑步", "")

    assert facts == ["用户喜欢跑步"]


async def test_extract_invalid_json_returns_empty(llm_service):
    """LLM 返回的不是 JSON → 解析失败，返回空（不抛异常）"""
    svc, fake_llm = llm_service
    fake_llm.content = "这不是JSON"

    facts = await svc.extract_facts("我叫张三", "")

    assert facts == []


async def test_extract_empty_response_returns_empty(llm_service):
    """LLM 返回空内容 → 返回空"""
    svc, fake_llm = llm_service
    fake_llm.content = ""

    facts = await svc.extract_facts("我叫张三", "")

    assert facts == []


async def test_extract_non_list_json_returns_empty(llm_service):
    """LLM 返回 JSON 但不是数组（比如对象）→ 返回空"""
    svc, fake_llm = llm_service
    fake_llm.content = '{"name": "张三"}'

    facts = await svc.extract_facts("我叫张三", "")

    assert facts == []


async def test_extract_only_uses_user_message(llm_service):
    """核心回归测试：萃取只基于用户消息，AI 回复不传给 LLM

    之前出过的 bug：萃取 LLM 会从 AI 回复里反推事实。
    比如用户只是问"有什么拍照好地方"，AI 回复里提到"你在东莞"，
    结果记忆里就多出一条"用户住在东莞"——这根本不是用户说的。

    本测试验证：
        1. 萃取结果确实来自用户消息
        2. 发给 LLM 的消息里只有 system + user，AI 回复一个字都不在
    """
    svc, fake_llm = llm_service
    user_msg = "我是东莞人，喜欢跑步"
    assistant_msg = "好的，推荐你去东莞的松山湖跑步"

    fake_llm.content = '["用户是东莞人", "用户喜欢跑步"]'
    facts = await svc.extract_facts(user_msg, assistant_msg)

    # 1. 萃取结果来自用户消息
    assert facts == ["用户是东莞人", "用户喜欢跑步"]

    # 2. LLM 只被调用了一次，且只收到 2 条消息
    assert len(fake_llm.calls) == 1
    messages = fake_llm.calls[0]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == FACT_EXTRACTION_PROMPT
    assert messages[1] == {"role": "user", "content": user_msg}

    # 3. AI 回复的内容绝不能出现在发给 LLM 的文本里
    all_content = " ".join(m["content"] for m in messages)
    assert assistant_msg not in all_content
    assert "松山湖" not in all_content
