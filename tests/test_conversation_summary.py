from __future__ import annotations

import json

import pytest

from agent.looping.ports import ContextConfig
from agent.provider import LLMResponse
from session.manager import SessionManager
from session.summary import ConversationSummaryService


class _SummaryProvider:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_after = fail_after

    async def chat(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("injected summary failure")
        payload = json.loads(kwargs["messages"][1]["content"])
        previous = payload["previous_summary"]
        goals = list(previous.get("user_goal") or [])
        goals.append(f"已压缩到 seq {payload['new_source_messages'][-1]['seq']}")
        return LLMResponse(
            content=json.dumps(
                {
                    "user_goal": goals,
                    "decisions": [],
                    "constraints": [],
                    "completed": [],
                    "artifacts": [],
                    "pending": ["继续后续对话"],
                    "important_results": [],
                },
                ensure_ascii=False,
            )
        )


async def _append_turns(manager: SessionManager, key: str, count: int) -> None:
    session = manager.get_or_create(key)
    start = len(session.messages)
    offset = start // 2
    for index in range(count):
        session.add_message("user", f"问题 {offset + index}")
        session.add_message("assistant", f"回答 {offset + index}")
    await manager.append_messages(session, session.messages[start:])


def _config(**overrides) -> ContextConfig:
    values = {
        "memory_window": 100,
        "auto_summary": True,
        "recent_turns": 2,
        "summary_trigger_messages": 12,
        "summary_min_new_messages": 4,
        "max_prompt_tokens": 16000,
        "summary_max_tokens": 500,
    }
    values.update(overrides)
    return ContextConfig(**values)


@pytest.mark.asyncio
async def test_summary_keeps_recent_turns_and_raw_messages(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    await _append_turns(manager, "api:a", 12)
    provider = _SummaryProvider()
    service = ConversationSummaryService(manager, provider, _config(), model="fast")

    assert await service.maybe_summarize("api:a") is True
    record = manager.get_summary("api:a")

    assert record is not None
    assert record["covered_from_seq"] == 0
    assert record["covered_through_seq"] == 19
    assert record["version"] == 1
    assert len(manager.get_or_create("api:a").messages) == 24
    assert manager._store.count_messages("api:a") == 24
    recent = service.history_for_prompt(
        manager.get_or_create("api:a"), max_messages=100
    )
    assert [item["content"] for item in recent] == [
        "问题 10",
        "回答 10",
        "问题 11",
        "回答 11",
    ]
    assert "seq 0..19" in service.get_prompt_context("api:a")
    manager.close()


@pytest.mark.asyncio
async def test_summary_is_incremental_and_survives_restart(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    await _append_turns(manager, "api:a", 12)
    provider = _SummaryProvider()
    service = ConversationSummaryService(manager, provider, _config())
    assert await service.maybe_summarize("api:a") is True

    await _append_turns(manager, "api:a", 4)
    assert await service.maybe_summarize("api:a") is True
    second = manager.get_summary("api:a")
    assert second is not None
    assert second["covered_through_seq"] == 27
    assert second["version"] == 2
    assert "已压缩到 seq 19" in json.loads(
        provider.calls[1]["messages"][1]["content"]
    )["previous_summary"]["user_goal"]
    manager.close()

    reopened = SessionManager(tmp_path)
    restored = reopened.get_summary("api:a")
    assert restored is not None
    assert restored["covered_through_seq"] == 27
    assert restored["version"] == 2
    assert len(reopened.get_or_create("api:a").messages) == 32
    reopened.close()


@pytest.mark.asyncio
async def test_summary_failure_does_not_advance_cursor(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    await _append_turns(manager, "api:a", 12)
    provider = _SummaryProvider(fail_after=1)
    service = ConversationSummaryService(manager, provider, _config())
    assert await service.maybe_summarize("api:a") is True
    before = manager.get_summary("api:a")

    await _append_turns(manager, "api:a", 4)
    assert await service.maybe_summarize("api:a") is False
    after = manager.get_summary("api:a")

    assert before is not None and after is not None
    assert after["covered_through_seq"] == before["covered_through_seq"]
    assert after["version"] == before["version"]
    assert len(manager.get_or_create("api:a").messages) == 32
    manager.close()


@pytest.mark.asyncio
async def test_emergency_budget_and_session_isolation(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    await _append_turns(manager, "api:a", 5)
    await _append_turns(manager, "api:b", 5)
    provider = _SummaryProvider()
    service = ConversationSummaryService(
        manager,
        provider,
        _config(summary_trigger_messages=999, max_prompt_tokens=10),
    )

    assert await service.ensure_budget(
        "api:a", current_message="继续", fixed_context="system"
    ) is True
    assert manager.get_summary("api:a") is not None
    assert manager.get_summary("api:b") is None
    assert len(manager.get_or_create("api:b").messages) == 10
    manager.close()
