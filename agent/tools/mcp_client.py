""" MCP 客户端 —— 连接远程 MCP Server，把远程工具注册到本地 ToolRegistry

使用方式：
    from agent.tools.mcp_client import MCPClient, register_mcp_server

    await register_mcp_server(
        registry,
        server_url="https://mcp.example.com",
        auth_token="sk-xxx",
    )
"""

from __future__ import annotations

import logging

import httpx

from agent.tools.base import BaseTool, ToolRegistry

logger = logging.getLogger(__name__)

# HTTP 超时设置（MCP Server 可能比较慢）
MCP_TIMEOUT = 30


# ── 第 1 层：HTTP 通信 ──────────────────────────────

class MCPClient:
    """MCP 客户端 —— 负责与远程 MCP Server 的 HTTP 通信

    职责：
        - list_tools()    → 获取 Server 上所有工具的定义
        - call_tool()     → 调用 Server 上的某个工具

    支持 REST 风格的 MCP 实现，也兼容标准 JSON-RPC 格式。
    """

    def __init__(self, server_url: str, auth_token: str = ""):
        self.server_url = server_url.rstrip("/")
        self.headers: dict[str, str] = {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

    async def list_tools(self) -> list[dict]:
        """获取远程 Server 上所有可用工具的定义

        返回格式：
            [
                {
                    "name": "get_weather",
                    "description": "...",
                    "inputSchema": {"type": "object", ...},
                },
                ...
            ]

        兼容两种 response 结构：
            - 直接返回 tools 数组
            - 返回 {"tools": [...]} 包裹格式
        """
        url = f"{self.server_url}/tools"
        logger.info("MCP 获取工具列表: %s", url)

        try:
            async with httpx.AsyncClient(timeout=MCP_TIMEOUT) as client:
                resp = await client.get(url, headers=self.headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            logger.error("MCP Server 超时: %s", url)
            return []
        except httpx.HTTPStatusError as e:
            logger.error("MCP Server 返回错误: %s", e)
            return []
        except Exception as e:
            logger.exception("MCP 连接失败: %s", url)
            return []

        # 兼容两种返回格式
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("tools", data.get("tool_list", []))
        return []

    async def call_tool(self, name: str, arguments: dict) -> str:
        """调用远程 Server 上的工具

        name:      工具名称
        arguments: 参数字典

        返回执行结果的文本。
        """
        url = f"{self.server_url}/call_tool"
        payload = {"name": name, "arguments": arguments}
        logger.info("MCP 调用工具: %s %s", name, arguments)

        try:
            async with httpx.AsyncClient(timeout=MCP_TIMEOUT) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers=self.headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            return f"工具 '{name}' 调用超时"
        except httpx.HTTPStatusError as e:
            return f"工具 '{name}' 返回错误 (HTTP {e.response.status_code})"
        except Exception as e:
            logger.exception("MCP 工具调用失败: %s", name)
            return f"工具 '{name}' 调用失败: {e}"

        # 兼容多种返回格式
        # MCP 标准: {"content": [{"type": "text", "text": "..."}], "isError": false}
        # 简化格式: {"result": "..."} 或直接返回字符串

        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            # MCP 标准格式：content 是数组
            if "content" in data:
                texts: list[str] = []
                for item in data["content"]:
                    if isinstance(item, dict):
                        texts.append(item.get("text", str(item)))
                    else:
                        texts.append(str(item))
                result = "\n".join(texts)
                if data.get("isError"):
                    return f"错误: {result}"
                return result

            # 简化格式
            if "result" in data:
                return str(data["result"])
            if "error" in data:
                return f"错误: {data['error']}"

            return str(data)

        return str(data)

    def __repr__(self) -> str:
        return f"MCPClient({self.server_url})"


# ── 第 2 层：适配 BaseTool ──────────────────────────

class MCPProxyTool(BaseTool):
    """远程 MCP 工具的本地代理

    对 ToolRegistry 和 agent_loop 来说，它就是一个普通 BaseTool。
    execute() 内部委托给 MCPClient 调远程 Server。
    """

    def __init__(self, client: MCPClient, meta: dict):
        """
        client: MCPClient 实例
        meta:   工具定义（来自 list_tools() 返回的每一项）
        """
        self._client = client
        self.name = meta.get("name", "unknown")
        self.description = meta.get("description", "")
        # inputSchema 是 MCP 标准字段，兼容 parameters
        self._params = meta.get("inputSchema") or meta.get("parameters") or {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @property
    def parameters(self) -> dict:
        return self._params

    async def execute(self, **kwargs) -> str:
        """委托 MCPClient 调远程 Server"""
        logger.info("MCP 代理执行: %s %s", self.name, kwargs)
        return await self._client.call_tool(self.name, kwargs)

    def __repr__(self) -> str:
        return f"MCPProxyTool(name={self.name!r}, via={self._client.server_url})"


# ── 第 3 层：集成辅助 ──────────────────────────────

async def register_mcp_server(
    registry: ToolRegistry,
    server_url: str,
    auth_token: str = "",
    prefix: str = "",
) -> int:
    """连接 MCP Server，将其所有工具注册到 ToolRegistry

    参数:
        registry:    ToolRegistry 实例
        server_url:  MCP Server 地址（如 https://mcp.example.com）
        auth_token:  认证令牌（可选）
        prefix:      工具名前缀（可选，防止不同 Server 工具名冲突）

    返回:
        成功注册的工具数量
    """
    client = MCPClient(server_url, auth_token)
    tools_meta = await client.list_tools()

    if not tools_meta:
        logger.warning("MCP Server 没有返回任何工具: %s", server_url)
        return 0

    count = 0
    for meta in tools_meta:
        name = meta.get("name", "")
        if not name:
            continue

        # 可选前缀
        full_name = f"{prefix}{name}" if prefix else name
        meta = {**meta, "name": full_name}

        tool = MCPProxyTool(client, meta)
        registry.register(
            tool,
            risk="external-side-effect",
            source_type="mcp",
            source_name=server_url,
        )
        count += 1

    logger.info(
        "MCP 同步完成: %s → %d 个工具已注册",
        server_url, count,
    )
    return count
