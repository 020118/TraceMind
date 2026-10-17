from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from agent.tools.code_execution import CheckPythonSyntaxTool, RunTestsTool


@pytest.fixture
def local_workspace() -> Path:
    root = Path(__file__).parent.parent / ".code_execution_testdata" / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_python_syntax_check_reports_success_and_error(local_workspace: Path) -> None:
    (local_workspace / "valid.py").write_text("answer = 42\n", encoding="utf-8")
    (local_workspace / "invalid.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    tool = CheckPythonSyntaxTool(local_workspace)

    valid = asyncio.run(tool.execute("valid.py"))
    invalid = asyncio.run(tool.execute("invalid.py"))

    assert "语法检查通过" in valid
    assert "语法检查未通过" in invalid
    assert "invalid.py:1" in invalid


def test_code_tools_reject_workspace_escape(local_workspace: Path) -> None:
    syntax = CheckPythonSyntaxTool(local_workspace)
    tests = RunTestsTool(local_workspace)

    syntax_result = asyncio.run(syntax.execute("../outside.py"))
    test_result = asyncio.run(tests.execute("../outside.py"))

    assert "超出工作目录" in syntax_result
    assert "超出工作目录" in test_result


def test_run_tests_reports_pass_and_failure(local_workspace: Path) -> None:
    (local_workspace / "test_pass.py").write_text(
        "def test_ok():\n    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )
    (local_workspace / "test_fail.py").write_text(
        "def test_bad():\n    assert 2 + 2 == 5\n",
        encoding="utf-8",
    )
    tool = RunTestsTool(local_workspace, max_timeout_seconds=10)

    passed = asyncio.run(tool.execute("test_pass.py", timeout_seconds=5))
    failed = asyncio.run(tool.execute("test_fail.py", timeout_seconds=5))

    assert "pytest 状态：通过" in passed
    assert "1 passed" in passed
    assert "pytest 状态：失败" in failed
    assert "1 failed" in failed


def test_run_tests_times_out_and_truncates_output(local_workspace: Path) -> None:
    (local_workspace / "test_slow.py").write_text(
        "import time\n\ndef test_slow():\n    time.sleep(5)\n",
        encoding="utf-8",
    )
    (local_workspace / "test_noisy.py").write_text(
        "def test_noisy():\n    print('x' * 30000)\n    assert False\n",
        encoding="utf-8",
    )
    tool = RunTestsTool(local_workspace, max_timeout_seconds=5)

    timed_out = asyncio.run(tool.execute("test_slow.py", timeout_seconds=1))
    noisy = asyncio.run(tool.execute("test_noisy.py", timeout_seconds=5))

    assert "pytest 状态：超时" in timed_out
    assert "进程已终止" in timed_out
    assert "输出已截断" in noisy
