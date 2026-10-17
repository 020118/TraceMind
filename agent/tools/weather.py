""" 天气查询工具 —— 调用 wttr.in 免费 API（无需 API Key）"""

from __future__ import annotations

import logging

import httpx

from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)

# wttr.in 免费天气 API（无需注册，无需 API Key）
WTTR_URL = "https://wttr.in"


class GetWeather(BaseTool):
    """查询指定城市的当前天气"""

    name = "get_weather"
    description = "获取指定城市的当前天气，包括温度、湿度、风速、天气状况等"
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名，支持中文，如 北京、上海、深圳、Tokyo、London",
            },
        },
        "required": ["city"],
    }

    async def execute(self, city: str) -> str:
        """调用 wttr.in 获取城市天气 JSON 数据"""
        logger.info("天气查询: %s", city)

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{WTTR_URL}/{city}",
                    params={"format": "j1", "lang": "zh"},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            return f"查询 {city} 天气超时，请稍后重试"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"未找到城市 '{city}'，请检查城市名是否正确"
            return f"天气服务暂时不可用 (HTTP {e.response.status_code})"
        except Exception as e:
            logger.exception("天气查询异常")
            return f"天气查询失败: {e}"

        return self._format_weather(city, data)

    # ── 内部辅助 ─────────────────────────────────────

    @staticmethod
    def _format_weather(city: str, data: dict) -> str:
        """把 API 返回的 JSON 转成可读文本"""
        try:
            current = data.get("current_condition", [{}])[0]
            desc = current.get("lang_zh", [{}])[0].get("value", "") or \
                   current.get("weatherDesc", [{}])[0].get("value", "")
            temp = current.get("temp_C", "?")
            feels = current.get("FeelsLikeC", "?")
            humidity = current.get("humidity", "?")
            wind = current.get("windspeedKmph", "?")
            wind_dir = current.get("winddir16Point", "?")
            uv = current.get("uvIndex", "?")
            vis = current.get("visibility", "?")

            separator = "=" * 30
            lines = [
                separator,
                f"  [天气] {city}",
                separator,
                f"  状况: {desc}",
                f"  温度: {temp}C（体感 {feels}C）",
                f"  湿度: {humidity}%",
                f"  风速: {wind} km/h ({wind_dir})",
                f"  UV指数: {uv}",
                f"  能见度: {vis} km",
            ]

            # 未来 3 天预报
            forecast = data.get("weather", [])[:3]
            if forecast:
                lines.append("")
                lines.append("  -- 未来 3 天预报 --")
                for day in forecast:
                    date = day.get("date", "?")
                    max_t = day.get("maxtempC", "?")
                    min_t = day.get("mintempC", "?")
                    desc_day = day.get("hourly", [{}])[0].get("lang_zh", [{}])[0].get("value", "")
                    lines.append(f"  {date}: {desc_day} {min_t}~{max_t}C")

            return "\n".join(lines)

        except (KeyError, IndexError, TypeError) as e:
            logger.warning("天气数据解析失败: %s", e)
            return f"{city} 天气数据获取成功但解析失败，原始数据: {str(data)[:200]}"
