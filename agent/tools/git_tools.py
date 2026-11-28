"""受控 Git 交付工具集（分支隔离、Diff 检查、人工确认后提交并推送）。

安全边界：
1. 所有命令均通过 asyncio.create_subprocess_exec("git", *args) 执行，绝不拼接 shell。
2. 禁止在 main / master 分支直接提交或推送，强制使用 agent/* 功能分支。
3. 禁止 --force 强推。
4. 提交前自动扫描暂存区与 diff，拦截敏感文件（.env / config.toml / *.pem / *.db）和真实密钥。
5. git_commit_and_push 标记 requires_confirmation=True，必须经过人工二次确认后才真正执行。
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from agent.security import check_sensitive_path, redact_secrets
from agent.tools.base import BaseTool, ToolRegistry

_PROTECTED_BRANCHES = frozenset({"main", "master", "HEAD", ""})
_MAX_DIFF_CHARS = 16_000
_GIT_TIMEOUT_SECONDS = 25.0


def _sanitize_branch_suffix(raw: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._/-]+", "-", str(raw or "").strip()).strip("-/.")
    clean = re.sub(r"/{2,}", "/", clean)
    clean = re.sub(r"\.{2,}", ".", clean)
    return clean[:60]


async def _run_git(
    workspace: Path,
    args: list[str],
    *,
    timeout: float = _GIT_TIMEOUT_SECONDS,
) -> tuple[int, str, str]:
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "USER", "LANG", "LC_ALL", "SSH_AUTH_SOCK", "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL")
        if key in os.environ
    }
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(workspace),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return -1, "", "未找到 git 可执行文件，请在运行环境中安装 git"
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", f"git 命令执行超时（>{timeout:g}s）"
    return (
        int(proc.returncode or 0),
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )


async def _current_branch(workspace: Path) -> str:
    rc, out, _ = await _run_git(workspace, ["rev-parse", "--abbrev-ref", "HEAD"])
    return out.strip() if rc == 0 else ""


class GitStatusTool(BaseTool):
    name = "git_status"
    description = "查看当前工作区的 Git 分支名称和文件改动状态（只读）。"
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.expanduser().resolve()

    async def execute(self, **_: Any) -> str:
        branch = await _current_branch(self._workspace)
        rc, out, err = await _run_git(self._workspace, ["status", "--short"])
        if rc != 0:
            return f"获取 Git 状态失败：{err.strip() or '当前目录可能不是 Git 仓库'}"
        body = out.strip() or "（工作区干净，没有未提交改动）"
        return f"当前分支：{branch or '未知'}\n改动状态：\n{body}"


class GitDiffTool(BaseTool):
    name = "git_diff"
    description = "查看当前工作区或暂存区的代码变更 diff（自动脱敏密钥并限制长度，只读）。"
    parameters = {
        "type": "object",
        "properties": {
            "staged": {
                "type": "boolean",
                "description": "是否仅查看已暂存（--cached）的改动，默认 false",
                "default": False,
            },
            "path": {
                "type": "string",
                "description": "可选，仅查看指定文件或目录的 diff",
            },
        },
        "required": [],
    }

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.expanduser().resolve()

    async def execute(self, staged: bool = False, path: str = "", **_: Any) -> str:
        args = ["diff"]
        if staged:
            args.append("--cached")
        clean_path = str(path or "").strip()
        if clean_path:
            args.extend(["--", clean_path])
        rc, out, err = await _run_git(self._workspace, args)
        if rc != 0:
            return f"获取 Git diff 失败：{err.strip()}"
        text = redact_secrets(out.strip())
        if not text:
            return "（当前没有可显示的 diff 改动）"
        if len(text) > _MAX_DIFF_CHARS:
            text = text[:_MAX_DIFF_CHARS] + f"\n\n... [diff 已截断，原始长度 {len(out)} 字符]"
        return text


class GitCreateBranchTool(BaseTool):
    name = "git_create_branch"
    description = (
        "新建并切换到一个隔离的 agent/* 功能分支，用于安全修改代码。"
        "不允许直接创建或切换到 main / master。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "功能分支名，例如 fix-login-bug（会自动规范化为 agent/fix-login-bug）",
            }
        },
        "required": ["name"],
    }

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.expanduser().resolve()

    async def execute(self, name: str, **_: Any) -> str:
        raw = str(name or "").strip()
        if raw.lower() in {"main", "master", "head"}:
            return "错误: 禁止创建或切换到受保护分支 main / master，请提供功能分支名。"
        suffix = raw[len("agent/"):] if raw.startswith("agent/") else raw
        safe = _sanitize_branch_suffix(suffix)
        if not safe:
            return "错误: 分支名无效，请使用字母、数字和短横线，例如 fix-timeout。"
        branch = f"agent/{safe}"

        rc, _, err = await _run_git(self._workspace, ["checkout", "-b", branch])
        if rc != 0:
            return f"创建分支失败：{err.strip()}"
        return f"已创建并切换到隔离功能分支：{branch}"


class GitCommitAndPushTool(BaseTool):
    name = "git_commit_and_push"
    description = (
        "在完成代码修改、自动化测试通过且独立 AI Review 通过后，将当前 agent/* 分支的改动提交并推送到远程仓库。"
        "该操作需要用户二次确认，且禁止直接推送到 main / master 或使用 --force。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "commit_message": {
                "type": "string",
                "description": "规范的 Git commit message，例如 feat: add timeout fallback",
            },
            "tests_passed": {
                "type": "boolean",
                "description": "是否已运行测试/构建并确认全部通过（必须为 true 才允许交付）",
            },
            "review_passed": {
                "type": "boolean",
                "description": "是否已调用 code_review_diff 并确认评审通过（必须为 true 才允许交付）",
            },
            "test_summary": {
                "type": "string",
                "description": "测试/构建结果摘要，将展示在人工确认弹窗中",
            },
            "review_summary": {
                "type": "string",
                "description": "AI Code Review 结论摘要，将展示在人工确认弹窗中",
            },
            "create_pr": {
                "type": "boolean",
                "description": "推送后是否尝试使用 gh CLI 自动创建 Pull Request，默认 false",
                "default": False,
            },
        },
        "required": [
            "commit_message",
            "tests_passed",
            "review_passed",
            "test_summary",
            "review_summary",
        ],
    }

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.expanduser().resolve()

    async def execute(
        self,
        commit_message: str,
        tests_passed: bool = False,
        review_passed: bool = False,
        test_summary: str = "",
        review_summary: str = "",
        create_pr: bool = False,
        _confirmed: bool = False,
        _operation_id: str = "",
        **_: Any,
    ) -> str:
        if not _confirmed:
            return "未执行：git_commit_and_push 必须经过用户明确二次确认后才允许执行。"
        if not tests_passed:
            return "未执行：测试未通过（tests_passed=false），已拒绝提交与推送。"
        if not review_passed:
            return "未执行：独立 AI Review 未通过（review_passed=false），请先完成 code_review_diff 并修复问题。"

        clean_msg = " ".join(str(commit_message or "").splitlines()).strip()
        if len(clean_msg) < 5:
            return "未执行：commit_message 过短，请提供清晰的提交说明。"

        branch = await _current_branch(self._workspace)
        if branch in _PROTECTED_BRANCHES:
            return (
                f"未执行：当前位于受保护分支 '{branch or '未知'}'，"
                "禁止直接提交推送。请先调用 git_create_branch 切换到 agent/* 分支。"
            )

        # 1. 检查工作区改动并拦截敏感文件
        rc, status_out, err = await _run_git(self._workspace, ["status", "--porcelain"])
        if rc != 0:
            return f"失败：无法读取 Git 状态：{err.strip()}"
        changed_lines = [line for line in status_out.splitlines() if line.strip()]
        if not changed_lines:
            return "未执行：当前工作区没有任何改动可提交。"

        for line in changed_lines:
            rel_str = line[3:].strip().strip('"')
            if " -> " in rel_str:
                rel_str = rel_str.split(" -> ", 1)[-1].strip().strip('"')
            try:
                check_sensitive_path(self._workspace / rel_str, self._workspace, for_write=True)
            except PermissionError as exc:
                return f"未执行：{exc}（文件：{rel_str}），已阻止提交。"

        # 2. 暂存并扫描 diff 中是否意外包含真实密钥
        rc, _, err = await _run_git(self._workspace, ["add", "-A"])
        if rc != 0:
            return f"失败：git add 失败：{err.strip()}"

        rc, staged_diff, _ = await _run_git(self._workspace, ["diff", "--cached"])
        if rc == 0 and redact_secrets(staged_diff) != staged_diff:
            await _run_git(self._workspace, ["reset", "HEAD"])
            return "未执行：在待提交的代码 diff 中检测到疑似真实 API Key 或私钥，已自动取消暂存并阻止推送。"

        # 3. 提交
        rc, commit_out, err = await _run_git(
            self._workspace, ["commit", "-m", clean_msg]
        )
        if rc != 0:
            return f"失败：git commit 失败：{err.strip() or commit_out.strip()}"

        # 4. 推送功能分支（绝不使用 --force）
        rc, push_out, push_err = await _run_git(
            self._workspace, ["push", "-u", "origin", branch], timeout=45.0
        )
        if rc != 0:
            return (
                f"本地已提交到分支 {branch}，但推送远程失败：{push_err.strip() or push_out.strip()}"
            )

        lines = [
            f"代码交付成功！",
            f"- 分支：{branch}",
            f"- 提交说明：{clean_msg}",
            f"- 测试摘要：{test_summary.strip() or '通过'}",
            f"- Review 摘要：{review_summary.strip() or '通过'}",
        ]

        # 5. 可选：尝试使用 gh CLI 创建 PR
        if create_pr:
            pr_note = await self._try_create_pr(branch, clean_msg, test_summary, review_summary)
            lines.append(f"- Pull Request：{pr_note}")

        return "\n".join(lines)

    async def _try_create_pr(
        self,
        branch: str,
        title: str,
        test_summary: str,
        review_summary: str,
    ) -> str:
        body = (
            f"## 变更摘要\n{title}\n\n"
            f"## 自动化测试\n{test_summary or '已通过'}\n\n"
            f"## AI Code Review\n{review_summary or '已通过'}\n"
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                "gh",
                "pr",
                "create",
                "--title",
                title,
                "--body",
                body,
                "--head",
                branch,
                cwd=str(self._workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=20.0)
            if proc.returncode == 0:
                return out_b.decode("utf-8", errors="replace").strip()
            return f"自动创建失败（可手动在 GitHub 网页创建）：{err_b.decode('utf-8', errors='replace').strip()}"
        except FileNotFoundError:
            return "本机未安装 gh CLI，请直接在 GitHub 仓库页面点击 Compare & pull request。"
        except Exception as exc:
            return f"PR 创建异常：{exc}"


def register_git_tools(registry: ToolRegistry, workspace: Path) -> None:
    """注册受控 Git 交付工具。"""
    resolved = workspace.expanduser().resolve()
    registry.register(
        GitStatusTool(resolved),
        risk="read-only",
        aliases=("git状态", "查看改动", "代码状态"),
    )
    registry.register(
        GitDiffTool(resolved),
        risk="read-only",
        aliases=("git diff", "查看代码差异", "改动详情"),
    )
    registry.register(
        GitCreateBranchTool(resolved),
        risk="write",
        aliases=("创建分支", "切换功能分支", "新建分支"),
    )
    registry.register(
        GitCommitAndPushTool(resolved),
        risk="destructive",
        requires_confirmation=True,
        aliases=("提交代码", "推送代码", "上传代码", "提PR", "交付代码"),
    )


__all__ = [
    "GitCommitAndPushTool",
    "GitCreateBranchTool",
    "GitDiffTool",
    "GitStatusTool",
    "register_git_tools",
]
