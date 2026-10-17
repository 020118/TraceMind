""" ToolRegistry 测试

运行方式：
    pytest tests/test_tool_registry.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from agent.tools.base import BaseTool, ToolRegistry


# 一个测试用工具
class FakeTool(BaseTool):
    name = "fake_tool"
    description = "测试用工具"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "文本"},
        },
        "required": ["text"],
    }

    async def execute(self, text: str = "") -> str:
        return f"fake: {text}"


# 一个没有 name 的坏工具
class BadTool(BaseTool):
    name = ""  # 故意留空
    description = "坏工具"

    async def execute(self, **kwargs) -> str:
        return ""


@pytest.fixture
def registry():
    return ToolRegistry()


# ── 注册 ────────────────────────────────────────────────

def test_register_increases_count(registry):
    """注册一个工具后 count 为 1"""
    registry.register(FakeTool())
    assert registry.count == 1


def test_register_and_get(registry):
    """注册后能通过 get 找到"""
    registry.register(FakeTool())
    tool = registry.get("fake_tool")
    assert tool is not None
    assert tool.name == "fake_tool"


# ── 异常场景 ────────────────────────────────────────────

def test_register_without_name_raises(registry):
    """注册没有 name 的工具应该抛 ValueError"""
    # TODO: 补全这个测试
    with pytest.raises(ValueError):
        registry.register(BadTool())


# ── 注销 ────────────────────────────────────────────────

def test_unregister(registry):
    """注销后 get 返回 None"""
    # TODO: 补全这个测试
    registry.register(FakeTool())
    registry.unregister("fake_tool")
    assert registry.get("fake_tool") is None
    assert registry.count == 0


# ── 查找 ────────────────────────────────────────────────

def test_get_nonexistent_returns_none(registry):
    """查找不存在的工具返回 None"""
    # TODO: 补全这个测试
    assert registry.get("不存在的") is None


# ── 格式转换 ────────────────────────────────────────────

def test_to_openai_tools_format(registry):
      """转成 OpenAI 格式"""
      registry.register(FakeTool())
      tools = registry.to_openai_tools()
      assert len(tools) == 1
      assert tools[0]["type"] == "function"
      assert tools[0]["function"]["name"] == "fake_tool"
      assert tools[0]["function"]["parameters"]["required"] == ["text"]


# ── 列表 ────────────────────────────────────────────────

def test_list_all(registry):
      """列出所有工具"""
      registry.register(FakeTool())
      tools = registry.list_all()
      assert len(tools) == 1
      assert tools[0].name == "fake_tool"


def test_search_indexes_tool_metadata(registry):
      """工具目录可以按名称、描述和别名搜索"""
      registry.register(FakeTool(), risk="write", aliases=("文本处理",))

      results = registry.search("文本处理")

      assert results[0]["name"] == "fake_tool"
      assert results[0]["risk"] == "write"


def test_registry_accepts_destructive_confirmation_metadata(registry):
      registry.register(FakeTool(), risk="destructive", requires_confirmation=True)

      metadata = registry.get_metadata("fake_tool")
      assert metadata is not None
      assert metadata.risk == "destructive"
      assert metadata.requires_confirmation is True


def test_deferred_registry_exposes_only_always_on_tools(registry):
      """延迟加载模式只暴露 always_on 工具"""
      registry.register(FakeTool())

      assert registry.get_initial_tool_names(deferred=True) == set()
      assert registry.get_initial_tool_names(deferred=False) == {"fake_tool"}
