from __future__ import annotations

import pytest

from agent.config_models import AgentConfig, LLMConfig, LLMProfile, RuntimeConfig
from agent.looping.core import AgentLoop
from agent.runtime import AgentRuntime


@pytest.mark.asyncio
async def test_runtime_starts_and_stops_without_external_channels(tmp_path) -> None:
    config = AgentConfig(
        llm=LLMConfig(
            provider="openai",
            main=LLMProfile(
                model="fake-model",
                api_key="test-key",
                base_url="http://127.0.0.1:1/v1",
            ),
        ),
        runtime=RuntimeConfig(workspace=str(tmp_path)),
        proactive={},
    )
    runtime = AgentRuntime(
        config,
        enable_console=False,
        enable_telegram=False,
    )

    await runtime.start()
    snapshot = runtime.status()

    assert runtime.ready is True
    assert snapshot["llm"] is True
    assert snapshot["telegram"] is False
    assert snapshot["proactive"] is False
    assert snapshot["agent_loop"] == "new"
    assert isinstance(runtime.agent_loop, AgentLoop)
    assert {
        "read_file",
        "list_dir",
        "edit_file",
        "write_file",
        "check_python_syntax",
        "run_tests",
        "tool_search",
    }.issubset({tool.name for tool in runtime.tools.list_all()})

    await runtime.stop()

    assert runtime.ready is False
    assert runtime.started is False
