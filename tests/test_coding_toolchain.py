from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from agent.tools.base import ToolRegistry
from agent.tools.code_execution import register_code_execution_tools
from agent.tools.filesystem import register_filesystem_tools
from agent.tools.tool_search import ToolSearchTool


@pytest.fixture
def coding_workspace() -> Path:
    root = Path(__file__).parent.parent / ".coding_toolchain_testdata" / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_coding_tools_close_the_modify_and_verify_loop(coding_workspace: Path) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry, coding_workspace)
    register_code_execution_tools(
        registry,
        coding_workspace,
        tool_timeout_seconds=10,
    )
    registry.register(ToolSearchTool(registry), always_on=True)

    write = registry.get("write_file")
    edit = registry.get("edit_file")
    syntax = registry.get("check_python_syntax")
    tests = registry.get("run_tests")
    search = registry.get("tool_search")
    assert write and edit and syntax and tests and search

    asyncio.run(
        write.execute(
            path="calculator.py",
            content="def add(left, right):\n    return left - right\n",
        )
    )
    asyncio.run(
        write.execute(
            path="test_calculator.py",
            content=(
                "from calculator import add\n\n"
                "def test_add():\n"
                "    assert add(2, 3) == 5\n"
            ),
        )
    )

    discovery = asyncio.run(search.execute(query="修改并测试 Python 代码"))
    first_test = asyncio.run(tests.execute(path="test_calculator.py", timeout_seconds=5))
    change = asyncio.run(
        edit.execute(
            path="calculator.py",
            old_text="return left - right",
            new_text="return left + right",
        )
    )
    syntax_result = asyncio.run(syntax.execute(path="calculator.py"))
    second_test = asyncio.run(tests.execute(path="test_calculator.py", timeout_seconds=5))

    assert "edit_file" in discovery
    assert "run_tests" in discovery
    assert "pytest 状态：失败" in first_test
    assert "-    return left - right" in change
    assert "+    return left + right" in change
    assert "语法检查通过" in syntax_result
    assert "pytest 状态：通过" in second_test
