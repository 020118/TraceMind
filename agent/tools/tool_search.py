"""工具目录搜索工具。"""

from __future__ import annotations

from typing import Any

from agent.tools.base import BaseTool, ToolRegistry


class ToolSearchTool(BaseTool):
    """让模型按任务描述发现当前注册的工具。"""

    name = "tool_search"
    description = (
        "搜索当前 Agent 已注册的工具。遇到不确定该使用什么工具、用户询问当前能力、"
        "或需要选择文件/网络/外部服务工具时，先调用本工具；query 留空可列出全部工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "任务描述或工具名称；留空表示列出所有当前可用工具",
            },
            "top_k": {
                "type": "integer",
                "description": "最多返回多少个匹配工具，默认 20，最大 30",
                "minimum": 1,
                "maximum": 30,
                "default": 20,
            },
            "risk": {
                "type": "string",
                "description": "可选风险过滤：read-only、write、destructive、external-side-effect",
                "enum": [
                    "read-only",
                    "write",
                    "destructive",
                    "external-side-effect",
                ],
            },
        },
        "required": [],
    }

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(
        self,
        query: str = "",
        top_k: int = 20,
        risk: str | None = None,
        **_: Any,
    ) -> str:
        results = self._registry.search(
            query,
            top_k=top_k,
            allowed_risk={risk} if risk else None,
        )
        if not results:
            return f"未找到匹配工具：{query!r}。请换一个任务描述，或将 query 留空查看全部工具。"

        lines = [f"工具目录搜索结果（{len(results)} 个）："]
        for index, result in enumerate(results, start=1):
            scope = "始终可用" if result["always_on"] else "需搜索后加载"
            confirmation = (
                "，需要用户确认" if result.get("requires_confirmation") else ""
            )
            source = result["source_type"]
            if result["source_name"]:
                source = f"{source}:{result['source_name']}"
            lines.append(
                f"{index}. {result['name']} "
                f"[{result['risk']}{confirmation}, {scope}, {source}]\n"
                f"   {result['description']}"
            )
        return "\n".join(lines)


__all__ = ["ToolSearchTool"]
