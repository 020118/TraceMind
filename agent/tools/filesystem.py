"""文件操作工具的 Agent Loop 适配层。

文件工具的核心实现仍位于 ``agent.host_tools.filesystem``，因为主动式
``proactive_v2`` 也在使用它们。本模块把这些工具适配为新 Agent Loop
使用的 ``BaseTool`` 接口，确保两套运行时共享同一套路径校验和编辑逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.host_tools.base import Tool as HostTool
from agent.host_tools.base import ToolResult as HostToolResult
from agent.host_tools.filesystem import (
    EditFileTool as HostEditFileTool,
    DeleteFileTool as HostDeleteFileTool,
    ListDirTool as HostListDirTool,
    ReadFileTool as HostReadFileTool,
    WriteFileTool as HostWriteFileTool,
)
from agent.tools.base import BaseTool, ToolRegistry


def _result_to_text(result: str | HostToolResult) -> str:
    """把旧工具可能返回的多模态结果转换为新循环可消费的文本。"""

    if isinstance(result, HostToolResult):
        return result.preview()
    return str(result)


class _FilesystemToolAdapter(BaseTool):
    """将一个旧 host tool 暴露为 BaseTool。"""

    def __init__(self, inner: HostTool) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def description(self) -> str:
        return self._inner.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._inner.parameters

    async def execute(self, **kwargs: Any) -> str:
        result = await self._inner.execute(**kwargs)
        return _result_to_text(result)


class ReadFileTool(_FilesystemToolAdapter):
    """在工作目录内读取文本或图片文件。"""

    def __init__(self, allowed_dir: Path) -> None:
        super().__init__(HostReadFileTool(allowed_dir=allowed_dir))


class WriteFileTool(_FilesystemToolAdapter):
    """在工作目录内创建文件或完整写入文件。"""

    def __init__(self, allowed_dir: Path) -> None:
        super().__init__(HostWriteFileTool(allowed_dir=allowed_dir))


class EditFileTool(_FilesystemToolAdapter):
    """在工作目录内精确替换文件内容并返回 diff。"""

    def __init__(self, allowed_dir: Path) -> None:
        super().__init__(HostEditFileTool(allowed_dir=allowed_dir))


class ListDirTool(_FilesystemToolAdapter):
    """列出工作目录内的文件和子目录。"""

    def __init__(self, allowed_dir: Path) -> None:
        super().__init__(HostListDirTool(allowed_dir=allowed_dir))


class DeleteFileTool(_FilesystemToolAdapter):
    """将 workspace 内的文件移入可恢复的回收目录。"""

    def __init__(self, allowed_dir: Path) -> None:
        super().__init__(HostDeleteFileTool(allowed_dir=allowed_dir))


def register_filesystem_tools(registry: ToolRegistry, workspace: Path) -> None:
    """将文件工具注册到新 Agent Loop。

    所有工具共享同一个 workspace 根目录。路径解析和越界校验由旧工具
    实现负责，因此 Agent 无法通过 ``..`` 访问 workspace 之外的文件。
    """

    allowed_dir = workspace.expanduser().resolve()
    allowed_dir.mkdir(parents=True, exist_ok=True)

    registry.register(
        ReadFileTool(allowed_dir),
        risk="read-only",
        aliases=("读取文件", "阅读代码", "查看代码"),
    )
    registry.register(
        ListDirTool(allowed_dir),
        risk="read-only",
        aliases=("查看目录", "项目结构", "文件列表"),
    )
    registry.register(
        EditFileTool(allowed_dir),
        risk="write",
        aliases=("修改文件", "修改代码", "编辑代码"),
    )
    registry.register(
        WriteFileTool(allowed_dir),
        risk="write",
        aliases=("创建文件", "写入文件", "生成代码"),
    )
    registry.register(
        DeleteFileTool(allowed_dir),
        risk="destructive",
        requires_confirmation=True,
        aliases=("删除文件", "移除文件", "清理文件"),
    )


__all__ = [
    "EditFileTool",
    "DeleteFileTool",
    "ListDirTool",
    "ReadFileTool",
    "WriteFileTool",
    "register_filesystem_tools",
]
