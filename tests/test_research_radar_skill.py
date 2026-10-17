"""内置 research-radar Drift Skill 的发现与状态闭环测试。"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from agent.skills import BUILTIN_SKILLS_DIR
from core.memory.engine import MemoryQueryResult, MemoryRecord
from proactive_v2.agent_tick_factory import BUILTIN_DRIFT_SKILL_NAMES
from proactive_v2.context import AgentTickContext
from proactive_v2.drift_state import DriftStateStore
from proactive_v2.drift_tools import (
    DriftToolDeps,
    FinishDriftTool,
    SelectSkillTool,
    build_drift_tool_registry,
)


def _make_store(tmp_path: Path) -> DriftStateStore:
    return DriftStateStore(
        tmp_path / "drift",
        builtin_skills_dir=BUILTIN_SKILLS_DIR,
        include_builtin_skills=True,
        builtin_skill_names=BUILTIN_DRIFT_SKILL_NAMES,
    )


@pytest.fixture
def local_tmp_path():
    """避开部分 Windows 环境中 pytest 系统临时目录的 ACL 问题。"""
    root = (
        Path(__file__).parent.parent
        / ".research_radar_testdata"
        / f"run-{uuid4().hex}"
    )
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_research_radar_is_discoverable_as_builtin_skill(local_tmp_path: Path) -> None:
    store = _make_store(local_tmp_path)

    skills = {skill.name: skill for skill in store.scan_skills()}

    assert "research-radar" in skills
    assert skills["research-radar"].builtin is True
    assert "研究兴趣" in skills["research-radar"].description


def test_research_radar_select_and_finish_persist_state(local_tmp_path: Path) -> None:
    store = _make_store(local_tmp_path)
    now = datetime(2026, 10, 2, 8, 30, tzinfo=timezone.utc)
    ctx = AgentTickContext(now_utc=now, session_key="test-session")

    selected_raw = asyncio.run(SelectSkillTool(ctx, store).execute("research-radar"))
    selected = json.loads(selected_raw)

    assert selected["ok"] is True
    assert "recent_seen_keys" in selected["content"]
    assert "message_push" in selected["content"]
    assert selected["local_context"]["cursor"] == {}

    finished_raw = asyncio.run(
        FinishDriftTool(ctx, store).execute(
            skill_used="research-radar",
            status="completed",
            briefing="完成 Agent reliability 主题检索，无需推送。",
            message_result="silent",
            cursor_update={
                "topic_index": 1,
                "last_topic": "agent reliability",
                "last_checked_at": now.isoformat(),
                "recent_seen_keys": ["arxiv:2609.01234"],
                "recent_pushed_keys": [],
            },
            journal_append=[
                {
                    "entry_type": "candidate_evaluated",
                    "key": "arxiv:2609.01234",
                    "payload": {
                        "topic": "agent reliability",
                        "score": 5,
                        "decision": "rejected",
                        "reason": "相关性不足",
                    },
                }
            ],
        )
    )

    assert json.loads(finished_raw) == {"ok": True}
    continuum = store.load_skill_continuum("research-radar")
    assert continuum["run_count"] == 1
    assert continuum["last_status"] == "completed"
    assert continuum["cursor"]["recent_seen_keys"] == ["arxiv:2609.01234"]

    journal = store.load_skill_journal("research-radar")
    assert len(journal) == 1
    assert journal[0]["entry_type"] == "candidate_evaluated"
    assert journal[0]["key"] == "arxiv:2609.01234"
    assert journal[0]["payload"]["decision"] == "rejected"


def test_drift_registers_runtime_memory_fallback(local_tmp_path: Path) -> None:
    class FakeMemory:
        def __init__(self) -> None:
            self.last_request = None

        async def query(self, request):
            self.last_request = request
            return MemoryQueryResult(
                text_block="- 用户正在研究 Agent 可靠性",
                records=[
                    MemoryRecord(
                        id="mem-1",
                        kind="fact",
                        summary="用户正在研究 Agent 可靠性",
                        score=0.9,
                        engine_kind="test",
                    )
                ],
            )

    store = _make_store(local_tmp_path)
    memory = FakeMemory()
    ctx = AgentTickContext(session_key="telegram:42")
    tools = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(
            drift_dir=local_tmp_path / "drift",
            store=store,
            memory=memory,
        ),
    )

    recall = tools.get_tool("recall_memory")
    assert recall is not None
    result = json.loads(asyncio.run(recall.execute(query="用户的研究方向")))

    assert result["hits"] == 1
    assert result["memory_ids"] == ["mem-1"]
    assert "Agent 可靠性" in result["result"]
    assert memory.last_request.intent == "interest"
    assert memory.last_request.effect == "read_only"
    assert memory.last_request.scope.channel == "telegram"
    assert memory.last_request.scope.chat_id == "42"


def test_runtime_memory_falls_back_to_long_term_profile(local_tmp_path: Path) -> None:
    class ProfileMemory:
        async def query(self, request):
            return MemoryQueryResult()

        def read_long_term(self):
            return "用户长期关注 Agent 记忆、可靠性与评测。"

    store = _make_store(local_tmp_path)
    ctx = AgentTickContext(session_key="api:researcher")
    tools = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(
            drift_dir=local_tmp_path / "drift",
            store=store,
            memory=ProfileMemory(),
        ),
    )

    recall = tools.get_tool("recall_memory")
    assert recall is not None
    result = json.loads(asyncio.run(recall.execute(query="用户的研究方向")))

    assert result["hits"] == 1
    assert result["memory_ids"] == ["long-term-profile"]
    assert "Agent 记忆" in result["result"]
