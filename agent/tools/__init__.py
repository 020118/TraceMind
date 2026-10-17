""" 工具系统 —— 让 LLM 能够调用外部函数

使用方式：
    from agent.tools.base import BaseTool, ToolRegistry

    class GetWeather(BaseTool):
        name = "get_weather"
        description = "获取指定城市的天气"
        parameters = {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"}
            },
            "required": ["city"],
        }
        async def execute(self, city: str) -> str:
            return f"{city} 的天气是 25°C"

    registry = ToolRegistry()
    registry.register(GetWeather())
"""

from agent.tools.base import BaseTool, ToolRegistry
