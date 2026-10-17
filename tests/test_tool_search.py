from __future__ import annotations

import pytest

from agent.tools.base import BaseTool, ToolRegistry
from agent.tools.tool_search import ToolSearchTool


class _CodeTool(BaseTool):
    name = "edit_code"
    description = "修改代码文件并返回变更 diff"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> str:
        return "edited"


@pytest.mark.asyncio
async def test_tool_search_returns_live_registry_metadata() -> None:
    registry = ToolRegistry()
    registry.register(_CodeTool(), risk="write", aliases=("代码修改",))
    registry.register(ToolSearchTool(registry), always_on=True)

    result = await registry.get("tool_search").execute(query="代码修改")

    assert "edit_code" in result
    assert "write" in result
    assert registry.get_initial_tool_names(deferred=True) == {"tool_search"}


@pytest.mark.asyncio
async def test_tool_search_empty_query_lists_current_tools() -> None:
    registry = ToolRegistry()
    registry.register(_CodeTool(), risk="write")
    registry.register(ToolSearchTool(registry), always_on=True)

    result = await registry.get("tool_search").execute(query="")

    assert "edit_code" in result
    assert "tool_search" in result
