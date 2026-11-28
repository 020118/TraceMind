from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git executable is required for delivery pipeline tests",
)

from agent.bus.queue import MessageBus
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig
from agent.provider import LLMResponse, ToolCall
from agent.tools.base import ToolRegistry
from agent.tools.code_review import CodeReviewDiffTool
from agent.tools.git_tools import (
    GitCommitAndPushTool,
    GitCreateBranchTool,
    GitDiffTool,
    GitStatusTool,
    register_git_tools,
)
from session.manager import SessionManager


def _init_repo_with_bare_remote(tmp_path: Path) -> tuple[Path, Path]:
    bare = tmp_path / "remote.git"
    bare.mkdir()
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)

    repo = tmp_path / "workspace"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "remote", "add", "origin", str(bare)], cwd=repo, check=True)
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=repo, check=True, capture_output=True)
    return repo, bare


@pytest.mark.asyncio
async def test_git_branch_and_commit_push_guardrails(tmp_path: Path):
    repo, bare = _init_repo_with_bare_remote(tmp_path)

    branch_tool = GitCreateBranchTool(repo)
    assert "禁止" in await branch_tool.execute(name="main")
    assert "禁止" in await branch_tool.execute(name="master")

    commit_tool = GitCommitAndPushTool(repo)
    # 1. Must be confirmed
    assert "未执行" in await commit_tool.execute(
        commit_message="feat: update app",
        tests_passed=True,
        review_passed=True,
        test_summary="all passed",
        review_summary="approved",
        _confirmed=False,
    )
    # 2. Must pass tests & review
    assert "测试未通过" in await commit_tool.execute(
        commit_message="feat: update app",
        tests_passed=False,
        review_passed=True,
        test_summary="failed",
        review_summary="approved",
        _confirmed=True,
    )
    assert "AI Review 未通过" in await commit_tool.execute(
        commit_message="feat: update app",
        tests_passed=True,
        review_passed=False,
        test_summary="passed",
        review_summary="rejected",
        _confirmed=True,
    )
    # 3. Blocked on protected branch main
    (repo / "app.py").write_text("def add(a, b):\n    return a + b + 0\n", encoding="utf-8")
    assert "受保护分支" in await commit_tool.execute(
        commit_message="feat: update app",
        tests_passed=True,
        review_passed=True,
        test_summary="passed",
        review_summary="approved",
        _confirmed=True,
    )

    # 4. Switch to agent/* branch
    res_branch = await branch_tool.execute(name="fix-add")
    assert "agent/fix-add" in res_branch

    # 5. Blocked if secret appears in diff
    (repo / "app.py").write_text(
        'API_KEY = "sk-1234567890abcdefghijklmnop"\n', encoding="utf-8"
    )
    res_secret = await commit_tool.execute(
        commit_message="feat: add key",
        tests_passed=True,
        review_passed=True,
        test_summary="passed",
        review_summary="approved",
        _confirmed=True,
    )
    assert "API Key" in res_secret

    # 6. Clean change commits and pushes to bare remote successfully
    (repo / "app.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
    status_text = await GitStatusTool(repo).execute()
    assert "agent/fix-add" in status_text
    diff_text = await GitDiffTool(repo).execute()
    assert "def add(a: int, b: int)" in diff_text

    res_ok = await commit_tool.execute(
        commit_message="refactor: add type annotations to add()",
        tests_passed=True,
        review_passed=True,
        test_summary="1 passed",
        review_summary="low risk, approved",
        _confirmed=True,
    )
    assert "代码交付成功" in res_ok
    remote_branches = subprocess.run(
        ["git", "branch"], cwd=bare, check=True, capture_output=True, text=True
    ).stdout
    assert "agent/fix-add" in remote_branches


@pytest.mark.asyncio
async def test_independent_code_review_diff_tool(tmp_path: Path):
    repo, _ = _init_repo_with_bare_remote(tmp_path)
    (repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

    seen_messages: list[list[dict]] = []

    class _RejectReviewerProvider:
        async def chat(self, **kwargs) -> LLMResponse:
            seen_messages.append(kwargs["messages"])
            return LLMResponse(
                content=json.dumps(
                    {
                        "approved": True,  # Even if LLM mistakenly sets True, high severity must force False
                        "risk_level": "high",
                        "summary": "发现逻辑回归错误",
                        "issues": [
                            {"severity": "high", "file": "app.py", "detail": "加法被改成了减法"}
                        ],
                        "suggestions": [],
                    },
                    ensure_ascii=False,
                )
            )

    review_tool = CodeReviewDiffTool(repo, _RejectReviewerProvider())
    report = await review_tool.execute(goal="优化 add 函数", test_summary="未运行")
    assert len(seen_messages) == 1
    assert len(seen_messages[0]) == 2  # Independent clean context: system + user only
    assert "未通过 (approved=false)" in report
    assert "加法被改成了减法" in report


@pytest.mark.asyncio
async def test_agent_loop_confirmation_flow_for_git_delivery(tmp_path: Path):
    repo, _ = _init_repo_with_bare_remote(tmp_path)
    subprocess.run(["git", "checkout", "-b", "agent/demo"], cwd=repo, check=True, capture_output=True)
    (repo / "app.py").write_text("def add(a, b):\n    return int(a) + int(b)\n", encoding="utf-8")

    bus = MessageBus()
    sm = SessionManager(repo)
    tools = ToolRegistry()
    register_git_tools(tools, repo)

    class _DeliveryProvider:
        async def chat(self, **kwargs) -> LLMResponse:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="git-1",
                        name="git_commit_and_push",
                        arguments={
                            "commit_message": "fix: coerce inputs to int",
                            "tests_passed": True,
                            "review_passed": True,
                            "test_summary": "pytest 1 passed",
                            "review_summary": "approved, low risk",
                        },
                    )
                ],
            )

    loop = AgentLoop(
        AgentLoopDeps(
            bus=bus,
            provider=_DeliveryProvider(),
            tools=tools,
            session_manager=sm,
            workspace=repo,
        ),
        AgentLoopConfig(llm=LLMConfig(model="m")),
    )
    try:
        # Step 1: Agent triggers git_commit_and_push -> paused for human confirmation
        r1 = await loop.process_direct("提交并推送代码", session_key="api:deliv1")
        assert "尚未执行" in r1
        pending = loop.reliability.get_pending_operation("api:deliv1")
        assert pending is not None
        assert pending.tool_name == "git_commit_and_push"

        # Step 2: User confirms -> executes commit & push
        r2 = await loop.process_direct("确认执行", session_key="api:deliv1")
        assert "代码交付成功" in r2
        assert loop.reliability.get_pending_operation("api:deliv1") is None
    finally:
        loop.close()
        sm.close()
