"""受控的 Python 代码检查与测试执行工具。

这里故意不提供任意 shell。所有命令由程序构造，并固定在 workspace 内执行。
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from agent.tools.base import BaseTool, ToolRegistry

_MAX_OUTPUT_BYTES = 20_000
_MAX_PYTHON_FILES = 500
_MAX_SOURCE_BYTES = 2_000_000


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_workspace_path(path: str, workspace: Path) -> Path:
    root = workspace.expanduser().resolve()
    candidate = Path(path).expanduser()
    resolved = (
        (root / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    if not _is_inside(resolved, root):
        raise PermissionError(f"路径 {path} 超出工作目录 {root}")
    return resolved


def _iter_python_files(target: Path, workspace: Path) -> list[Path]:
    if not target.exists():
        raise FileNotFoundError(f"路径不存在：{target}")
    if target.is_file():
        if target.suffix.lower() != ".py":
            raise ValueError("语法检查只支持 .py 文件或目录")
        return [target]
    if not target.is_dir():
        raise ValueError(f"目标不是文件或目录：{target}")

    files: list[Path] = []
    for candidate in target.rglob("*.py"):
        if any(part in {".git", ".venv", "venv", "__pycache__"} for part in candidate.parts):
            continue
        resolved = candidate.resolve()
        if not _is_inside(resolved, workspace):
            continue
        files.append(resolved)
        if len(files) > _MAX_PYTHON_FILES:
            raise ValueError(f"Python 文件超过 {_MAX_PYTHON_FILES} 个，请缩小检查范围")
    return sorted(files)


class CheckPythonSyntaxTool(BaseTool):
    """使用 Python compile 检查源码，不执行用户代码。"""

    name = "check_python_syntax"
    description = (
        "检查 workspace 内一个 Python 文件或目录的语法，不执行代码。"
        "修改 Python 代码后应先使用本工具，再运行测试。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要检查的 .py 文件或目录，相对路径基于 workspace",
            }
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.expanduser().resolve()

    async def execute(self, path: str, **_: Any) -> str:
        try:
            target = _resolve_workspace_path(path, self._workspace)
            files = _iter_python_files(target, self._workspace)
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            return f"Python 语法检查失败：{exc}"

        if not files:
            return f"Python 语法检查失败：{path} 中没有找到 .py 文件"

        errors: list[str] = []
        checked = 0
        for file_path in files:
            try:
                if file_path.stat().st_size > _MAX_SOURCE_BYTES:
                    errors.append(
                        f"{file_path.relative_to(self._workspace)}：文件超过 2MB，已拒绝检查"
                    )
                    continue
                source = file_path.read_text(encoding="utf-8-sig")
                compile(source, str(file_path), "exec")
                checked += 1
            except SyntaxError as exc:
                relative = file_path.relative_to(self._workspace)
                line = (exc.text or "").strip()
                errors.append(
                    f"{relative}:{exc.lineno or '?'}:{exc.offset or '?'} "
                    f"{exc.msg}" + (f"\n  {line}" if line else "")
                )
            except (OSError, UnicodeError) as exc:
                errors.append(f"{file_path.relative_to(self._workspace)}：{exc}")

        if errors:
            return (
                f"Python 语法检查未通过：检查 {len(files)} 个文件，"
                f"发现 {len(errors)} 个问题。\n" + "\n".join(errors)
            )
        return f"Python 语法检查通过：共检查 {checked} 个文件。"


class RunTestsTool(BaseTool):
    """以固定的 pytest 命令运行 workspace 内的测试。"""

    name = "run_tests"
    description = (
        "在 workspace 内运行指定 pytest 测试。命令不可自定义，具有路径限制、"
        "超时、失败数量和输出长度限制。测试代码会被真正执行，可能产生副作用。"
    )

    def __init__(self, workspace: Path, max_timeout_seconds: float = 29.0) -> None:
        self._workspace = workspace.expanduser().resolve()
        self._max_timeout = max(1.0, float(max_timeout_seconds))

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "测试文件或测试目录，相对路径基于 workspace",
                },
                "keyword": {
                    "type": "string",
                    "description": "可选 pytest -k 过滤表达式，不作为 shell 命令执行",
                    "maxLength": 200,
                },
                "max_failures": {
                    "type": "integer",
                    "description": "最多报告多少个失败，默认 3，范围 1-10",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": f"执行超时，默认不超过 {self._max_timeout:g} 秒",
                    "minimum": 1,
                    "maximum": self._max_timeout,
                    "default": min(25.0, self._max_timeout),
                },
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        keyword: str = "",
        max_failures: int = 3,
        timeout_seconds: float | None = None,
        **_: Any,
    ) -> str:
        try:
            target = _resolve_workspace_path(path, self._workspace)
        except PermissionError as exc:
            return f"测试执行被拒绝：{exc}"
        if not target.exists():
            return f"测试执行失败：路径不存在：{path}"
        if not (target.is_file() or target.is_dir()):
            return f"测试执行失败：目标不是文件或目录：{path}"

        try:
            failures = max(1, min(int(max_failures), 10))
        except (TypeError, ValueError):
            failures = 3
        try:
            requested_timeout = (
                min(25.0, self._max_timeout)
                if timeout_seconds is None
                else float(timeout_seconds)
            )
        except (TypeError, ValueError):
            requested_timeout = min(25.0, self._max_timeout)
        timeout = max(1.0, min(requested_timeout, self._max_timeout))
        keyword = str(keyword).strip()[:200]

        relative_target = str(target.relative_to(self._workspace)) or "."
        command = [
            sys.executable,
            "-m",
            "pytest",
            relative_target,
            "-q",
            "--color=no",
            "--disable-warnings",
            "-p",
            "no:cacheprovider",
            "--maxfail",
            str(failures),
        ]
        if keyword:
            command.extend(["-k", keyword])

        return await _run_bounded_process(
            command,
            cwd=self._workspace,
            timeout_seconds=timeout,
            output_limit=_MAX_OUTPUT_BYTES,
        )


def _sanitized_environment(workspace: Path) -> dict[str, str]:
    """不给测试子进程继承 API key、Token、代理等 Agent 凭据。"""
    allowed = (
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "LANG",
        "LC_ALL",
    )
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    sandbox = workspace / ".agent_test_runtime"
    sandbox_home = sandbox / "home"
    sandbox_tmp = sandbox / "tmp"
    sandbox_home.mkdir(parents=True, exist_ok=True)
    sandbox_tmp.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(sandbox_home),
            "USERPROFILE": str(sandbox_home),
            "TEMP": str(sandbox_tmp),
            "TMP": str(sandbox_tmp),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "NO_COLOR": "1",
        }
    )
    return env


async def _run_bounded_process(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    output_limit: int,
) -> str:
    started = time.monotonic()
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=_sanitized_environment(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=(os.name != "nt"),
        )
    except (OSError, ValueError) as exc:
        return f"测试进程启动失败：{exc}"

    output_task = asyncio.create_task(
        _read_limited_output(process.stdout, output_limit)
    )
    timed_out = False
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        timed_out = True
        _kill_process(process)
        await process.wait()
    except asyncio.CancelledError:
        _kill_process(process)
        await asyncio.shield(process.wait())
        output_task.cancel()
        await asyncio.gather(output_task, return_exceptions=True)
        raise

    try:
        output, total_bytes, truncated = await asyncio.wait_for(output_task, timeout=2)
    except asyncio.TimeoutError:
        output_task.cancel()
        await asyncio.gather(output_task, return_exceptions=True)
        output, total_bytes, truncated = "[输出管道未及时关闭]", 0, True

    elapsed = time.monotonic() - started
    if timed_out:
        status = f"超时（限制 {timeout_seconds:g} 秒，进程已终止）"
    elif process.returncode == 0:
        status = "通过"
    else:
        status = "失败"
    truncation = (
        f"\n[输出已截断：原始输出约 {total_bytes} 字节，保留 {output_limit} 字节]"
        if truncated
        else ""
    )
    return (
        f"pytest 状态：{status}\n"
        f"退出码：{process.returncode}\n"
        f"耗时：{elapsed:.2f} 秒\n"
        f"执行目标：{command[3]}\n\n"
        f"{output.strip() or '[pytest 没有输出]'}{truncation}"
    )


async def _read_limited_output(
    stream: asyncio.StreamReader | None,
    limit: int,
) -> tuple[str, int, bool]:
    if stream is None:
        return "", 0, False
    head_limit = max(1, (limit * 2) // 3)
    tail_limit = max(1, limit - head_limit)
    head = bytearray()
    tail = bytearray()
    total = 0

    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        total += len(chunk)
        remaining_head = head_limit - len(head)
        if remaining_head > 0:
            head.extend(chunk[:remaining_head])
            chunk = chunk[remaining_head:]
        if chunk:
            tail.extend(chunk)
            if len(tail) > tail_limit:
                del tail[:-tail_limit]

    truncated = total > len(head) + len(tail)
    separator = b"\n... [middle output truncated] ...\n" if truncated else b""
    combined = bytes(head) + separator + bytes(tail)
    return combined.decode("utf-8", errors="replace"), total, truncated


def _kill_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError):
        pass


def register_code_execution_tools(
    registry: ToolRegistry,
    workspace: Path,
    *,
    tool_timeout_seconds: float = 30.0,
) -> None:
    """注册代码验证工具，保留一秒给 Agent Loop 做超时收尾。"""
    process_timeout = max(1.0, float(tool_timeout_seconds) - 1.0)
    registry.register(
        CheckPythonSyntaxTool(workspace),
        risk="read-only",
        aliases=("Python语法检查", "编译检查", "检查代码"),
    )
    registry.register(
        RunTestsTool(workspace, max_timeout_seconds=process_timeout),
        risk="external-side-effect",
        aliases=("运行测试", "pytest", "验证代码", "测试代码"),
    )


__all__ = [
    "CheckPythonSyntaxTool",
    "RunTestsTool",
    "register_code_execution_tools",
]
