""" 工具系统核心 —— BaseTool 抽象基类 + ToolRegistry 注册表 """

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolMetadata:
    """工具目录中的元数据。"""

    risk: str = "read-only"
    always_on: bool = False
    aliases: tuple[str, ...] = ()
    source_type: str = "builtin"
    source_name: str = ""
    requires_confirmation: bool = False


class BaseTool(ABC):
    """工具抽象基类

    子类需定义:
        name: str        — 工具名称（LLM 通过此名称调用）
        description: str — 工具描述（LLM 判断何时使用此工具）
        execute()        — 工具的实际执行逻辑

    可选定义:
        parameters: dict — 参数的 JSON Schema（无参工具可省略）
    """

    name: str = ""
    description: str = ""

    @property
    def parameters(self) -> dict:
        """参数 JSON Schema，默认无参数

        格式遵循 OpenAI function calling 的 parameters 规范：
            {
                "type": "object",
                "properties": {
                    "param_name": {
                        "type": "string",
                        "description": "参数说明"
                    }
                },
                "required": ["param_name"]
            }
        """
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """执行工具逻辑

        返回值会以字符串形式送回给 LLM 作为工具调用结果。
        如果执行失败，应抛出异常，框架会自动捕获并告知 LLM。
        """
        ...

    # ── 格式转换 ─────────────────────────────────────

    def to_openai_tool(self) -> dict:
        """转为 OpenAI tools API 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_mcp_format(self) -> dict:
        """转为 MCP (Model Context Protocol) 格式"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    # ── 辅助 ──────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class ToolRegistry:
    """工具注册表 —— 管理工具的注册、查找和格式转换

    职责：
        - register() / unregister()   — 运行时增删工具
        - get() / list_all()          — 查询已注册工具
        - to_openai_tools()           — 转成 LLM API 所需的格式
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._metadata: dict[str, ToolMetadata] = {}

    def register(
        self,
        tool: BaseTool,
        *,
        risk: str = "read-only",
        always_on: bool = False,
        aliases: tuple[str, ...] | list[str] = (),
        source_type: str = "builtin",
        source_name: str = "",
        requires_confirmation: bool = False,
    ) -> None:
        """注册一个工具实例"""
        if not tool.name:
            raise ValueError(f"工具缺少 name 属性: {type(tool).__name__}")
        if risk not in {
            "read-only",
            "write",
            "destructive",
            "external-side-effect",
        }:
            raise ValueError(f"不支持的工具风险等级: {risk}")
        if tool.name in self._tools:
            logger.warning("工具 '%s' 已存在，将被覆盖", tool.name)
        self._tools[tool.name] = tool
        self._metadata[tool.name] = ToolMetadata(
            risk=risk,
            always_on=always_on,
            aliases=tuple(aliases),
            source_type=source_type,
            source_name=source_name,
            requires_confirmation=requires_confirmation,
        )
        logger.info("工具已注册: %s — %s", tool.name, tool.description)

    def unregister(self, name: str) -> None:
        """注销指定名称的工具"""
        if name in self._tools:
            del self._tools[name]
            self._metadata.pop(name, None)
            logger.info("工具已注销: %s", name)

    def get(self, name: str) -> BaseTool | None:
        """按名称查找工具，未找到返回 None"""
        return self._tools.get(name)

    def list_all(self) -> list[BaseTool]:
        """列出所有已注册的工具"""
        return list(self._tools.values())

    def to_openai_tools(
        self, names: set[str] | list[str] | None = None
    ) -> list[dict]:
        """将指定工具转为 LLM API 格式；不传 names 时返回全部工具。"""
        if names is None:
            tools = self._tools.values()
        else:
            selected = set(names)
            tools = (tool for name, tool in self._tools.items() if name in selected)
        return [tool.to_openai_tool() for tool in tools]

    def get_metadata(self, name: str) -> ToolMetadata | None:
        return self._metadata.get(name)

    def get_always_on_names(self) -> set[str]:
        return {
            name for name, metadata in self._metadata.items() if metadata.always_on
        }

    def get_initial_tool_names(self, *, deferred: bool = False) -> set[str]:
        """返回一轮推理开始时暴露给模型的工具名称。"""
        if not deferred:
            return set(self._tools)
        return self.get_always_on_names()

    def search(
        self,
        query: str = "",
        *,
        top_k: int = 8,
        allowed_risk: set[str] | None = None,
        include_always_on: bool = True,
    ) -> list[dict[str, Any]]:
        """按名称、描述和别名搜索工具目录。"""
        top_k = max(1, min(int(top_k), 30))
        risk_filter = allowed_risk or set()
        tokens = _tool_search_tokens(query)
        results: list[tuple[int, str, dict[str, Any]]] = []

        for name, tool in self._tools.items():
            metadata = self._metadata.get(name, ToolMetadata())
            if risk_filter and metadata.risk not in risk_filter:
                continue
            if not include_always_on and metadata.always_on:
                continue

            score = 1 if not tokens else _tool_search_score(
                name, tool.description, metadata.aliases, tokens
            )
            if score <= 0:
                continue
            results.append(
                (
                    score,
                    name,
                    {
                        "name": name,
                        "description": tool.description,
                        "risk": metadata.risk,
                        "always_on": metadata.always_on,
                        "source_type": metadata.source_type,
                        "source_name": metadata.source_name,
                        "requires_confirmation": metadata.requires_confirmation,
                    },
                )
            )

        results.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in results[:top_k]]

    @property
    def count(self) -> int:
        """已注册的工具数量"""
        return len(self._tools)

    def __repr__(self) -> str:
        names = ", ".join(self._tools.keys())
        return f"ToolRegistry({self.count} tools: [{names}])"


def _tool_search_tokens(query: str) -> set[str]:
    normalized = query.lower().strip()
    if not normalized:
        return set()
    tokens: set[str] = {normalized}
    for part in normalized.split():
        tokens.add(part)
    for segment in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", normalized):
        tokens.add(segment)
        if all("\u4e00" <= char <= "\u9fff" for char in segment):
            tokens.update(segment)
            tokens.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return {token for token in tokens if token}


def _tool_search_score(
    name: str,
    description: str,
    aliases: tuple[str, ...],
    tokens: set[str],
) -> int:
    name_lower = name.lower()
    name_parts = [part for part in name_lower.split("_") if part]
    description_lower = description.lower()
    aliases_lower = " ".join(aliases).lower()
    score = 0
    for token in tokens:
        if token in name_parts:
            score += 10
        elif token in name_lower:
            score += 5
        if token in aliases_lower:
            score += 4
        if token in description_lower:
            score += 2
    return score
