""" MCP stdio 客户端（方案B）—— 通过子进程连接标准 MCP Server

与 mcp_client.py 的区别：
    mcp_client.py           → 自定义 REST 协议（自己定义的 HTTP 接口）
    mcp_stdio_client.py     → 标准 MCP 协议（JSON-RPC over stdio）

使用方式：
    # 1. 启动 MCP Server 子进程，注册所有工具
    server = await MCPStdioServer.create(
        command="uvx",
        args=["amap-mcp-server"],
        env={"AMAP_MAPS_API_KEY": "你的Key"},
    )
    count = await server.register_tools(registry)

    # 2. 使用完毕时关闭
    await server.close()

    # 3. 快捷方式：一行注册
    count = await register_mcp_stdio(
        registry,
        command="uvx",
        args=["amap-mcp-server"],
        env={"AMAP_MAPS_API_KEY": "你的Key"},
    )
"""

from __future__ import annotations

import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.tools.base import BaseTool, ToolRegistry

logger = logging.getLogger(__name__)


# ── MCP 工具代理 ──────────────────────────────────────

class MCPStdioProxyTool(BaseTool):
    """标准 MCP 协议的远程工具代理

    通过 ClientSession 调用远程 MCP Server 的工具。
    与 MCPProxyTool（mcp_client.py）的区别：
        - 使用 JSON-RPC over stdio（标准 MCP 协议）
        - 生命周期由 MCPStdioServer 管理
    """

    def __init__(self, session: ClientSession, tool_meta: dict):
        """
        session:    已初始化的 MCP ClientSession
        tool_meta:  工具定义字典，包含 name/description/inputSchema
        """
        self._session = session
        self.name = tool_meta.get("name", "unknown")
        self.description = tool_meta.get("description", "")
        self._params = (
            tool_meta.get("inputSchema")
            or tool_meta.get("input_schema")
            or {"type": "object", "properties": {}}
        )

    @property
    def parameters(self) -> dict:
        return self._params

    async def execute(self, **kwargs: Any) -> str:
        """通过 MCP JSON-RPC 调用远程工具"""
        logger.info("MCP(stdio) 调用: %s %s", self.name, kwargs)
        try:
            result = await self._session.call_tool(self.name, kwargs)
        except Exception as e:
            logger.exception("MCP(stdio) 工具调用失败: %s", self.name)
            return f"工具 '{self.name}' 调用失败: {e}"

        # 解析标准 CallToolResult 的 content 数组
        if result.isError:
            texts = [_extract_text(item) for item in result.content]
            return f"错误: {'; '.join(texts)}"

        texts = [_extract_text(item) for item in result.content]
        return "\n".join(texts)

    def __repr__(self) -> str:
        return f"MCPStdioProxyTool(name={self.name!r})"


def _extract_text(item: Any) -> str:
    """从 MCP content item 中提取文本"""
    if hasattr(item, "text"):
        return item.text
    if isinstance(item, dict):
        return item.get("text", str(item))
    return str(item)


# ── MCP Server 生命周期管理 ────────────────────────────

class MCPStdioServer:
    """管理一个 MCP Server 的子进程生命周期

    用法：
        server = await MCPStdioServer.create("uvx", ["amap-mcp-server"], env={...})
        try:
            await server.register_tools(registry)
            # ... agent 运行期间保持 server 存活 ...
        finally:
            await server.close()
    """

    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._ctx = None  # 持有 stdio_client 的 context manager
        self._read_write = None

    @classmethod
    async def create(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> MCPStdioServer:
        """创建并初始化 MCP 连接

        参数:
            command: 可执行命令，如 "uvx"
            args:    命令行参数，如 ["amap-mcp-server"]
            env:     环境变量，如 {"AMAP_MAPS_API_KEY": "xxx"}
        """
        self = cls()
        server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )

        # 进入 stdio_client 上下文，建立子进程连接
        ctx = stdio_client(server_params)
        self._ctx = ctx
        read, write = await ctx.__aenter__()  # 这里手动调__aenter__的作用是创建子进程，如果使用async with在退出时自动关闭资源，而我们需要在Agent在运行的整个过程中保持连接所以手动进入上下文，手动控制退出时机
        self._read_write = (read, write)

        # 创建并初始化 MCP 会话
        session = await ClientSession(read, write).__aenter__()  # 这里手动调__aenter__的作用是创建session会话，手动创建原因如上
        self._session = session
        await session.initialize()



        server_info = session.server_capabilities or {}
        logger.info(
            "MCP Server 已连接: %s %s | %s",
            command, " ".join(args or []),
            server_info,
        )
        return self

    async def list_tools(self) -> list[dict]:
        """获取 MCP Server 的所有工具定义"""
        if not self._session:
            logger.error("MCP Server 未初始化")
            return []

        try:
            result = await self._session.list_tools()
            raw_tools = getattr(result, "tools", result) if hasattr(result, "tools") else result
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema or {"type": "object", "properties": {}},
                }
                for t in raw_tools
            ]
        except Exception as e:
            logger.exception("MCP list_tools 失败")
            return []

    async def register_tools(
        self,
        registry: ToolRegistry,
        prefix: str = "",
    ) -> int:
        """将该 MCP Server 的所有工具注册到 ToolRegistry

        参数:
            registry: ToolRegistry 实例
            prefix:   工具名前缀（可选）

        返回:
            成功注册的工具数量
        """
        if not self._session:
            logger.error("MCP Server 未初始化，无法注册工具")
            return 0

        tools_meta = await self.list_tools()
        if not tools_meta:
            logger.warning("MCP Server 没有返回任何工具")
            return 0

        count = 0
        for meta in tools_meta:
            name = meta.get("name", "")
            if not name:
                continue

            if prefix:
                meta = {**meta, "name": f"{prefix}{name}"}

            tool = MCPStdioProxyTool(self._session, meta)
            registry.register(
                tool,
                risk="external-side-effect",
                source_type="mcp",
                source_name="stdio",
            )
            count += 1

        logger.info(
            "MCP(stdio) 同步完成: %d 个工具已注册", count,
        )
        return count

    async def close(self) -> None:
        """关闭 MCP 连接，终止子进程"""
        errors = []

        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception as e:
                errors.append(f"session关闭失败: {e}")
            self._session = None

        if self._ctx:
            try:
                await self._ctx.__aexit__(None, None, None)
            except Exception as e:
                errors.append(f"子进程关闭失败: {e}")
            self._ctx = None
            self._read_write = None

        if errors:
            logger.warning("MCP Server 关闭时出现错误: %s", "; ".join(errors))
        else:
            logger.info("MCP Server 已关闭")

    async def __aenter__(self) -> MCPStdioServer:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"MCPStdioServer(session={'alive' if self._session else 'closed'})"


# ── 快捷注册函数 ──────────────────────────────────────

async def register_mcp_stdio(
    registry: ToolRegistry,
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    prefix: str = "",
) -> MCPStdioServer:
    """快捷方式：连接 MCP Server 并注册所有工具

    返回 MCPStdioServer 实例，调用方需要在程序退出时 close()。

    示例：
        server = await register_mcp_stdio(
            registry,
            command="uvx",
            args=["amap-mcp-server"],
            env={"AMAP_MAPS_API_KEY": "xxx"},
        )
        # ... 运行 agent ...
        await server.close()
    """
    server = await MCPStdioServer.create(command, args, env)
    await server.register_tools(registry, prefix=prefix)
    return server
