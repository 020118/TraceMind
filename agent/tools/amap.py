""" 高德地图工具（方案A）—— 直接调用高德 REST API    工具来源：https://github.com/sugarforever/amap-mcp-server

使用方式：
    from agent.tools.amap import AmapGeo, AmapWeather, AmapTextSearch, ...

    工具列表：
        amap_geo             地理编码（地址→经纬度）
        amap_regeo           逆地理编码（经纬度→地址）
        amap_weather         天气查询
        amap_text_search     POI 关键词搜索
        amap_around_search   POI 周边搜索
        amap_search_detail   POI 详情查询
        amap_direction       驾车路线规划
        amap_distance        距离测量

配置：
    设置环境变量 AMAP_API_KEY，或者在实例化时传入 api_key。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────

API_KEY = os.environ.get("AMAP_API_KEY", "")
AMAP_BASE = "https://restapi.amap.com/v3"


# ── 公共辅助 ──────────────────────────────────────────

class AmapClient:
    """高德 API HTTP 客户端封装

    统一处理：超时、HTTP 错误、业务状态码判断、异常转友好消息。
    """

    TIMEOUT = 15

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or API_KEY
        if not self._api_key:
            raise ValueError(
                "缺少高德 API Key。请在环境变量 AMAP_API_KEY 中设置，"
                "或在实例化时传入 api_key 参数。"
            )

    async def get(self, path: str, params: dict | None = None) -> dict:
        """发起 GET 请求，返回高德 API 标准响应 JSON"""
        url = f"{AMAP_BASE}{path}"
        params = {"key": self._api_key, **(params or {})}

        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            return {"status": "0", "info": f"请求超时: {url}"}
        except httpx.HTTPStatusError as e:
            return {"status": "0", "info": f"HTTP {e.response.status_code}"}
        except Exception as e:
            logger.exception("高德 API 请求失败: %s", url)
            return {"status": "0", "info": str(e)}

        # 高德返回 status=0 表示业务错误
        if data.get("status") != "1":
            info = data.get("info", "未知错误")
            logger.warning("高德 API 返回错误: %s (%s)", info, url)
            return {"status": "0", "info": info}

        return data

    @staticmethod
    def _check_api_key(data: dict) -> bool:
        """如果 API Key 无效，记录警告"""
        if data.get("info") in ("INVALID_USER_KEY", "USER_IS_NOT_VALID"):
            logger.error("高德 API Key 无效: %s", data.get("info"))
            return False
        return True


# ── 1. 地理编码 ──────────────────────────────────────

class AmapGeo(BaseTool):
    """地理编码：将结构化的地址转换为经纬度坐标"""

    name = "amap_geo"
    description = "将详细地址转换为经纬度坐标（高德地图地理编码）"
    parameters = {
        "type": "object",
        "properties": {
            "address": {
                "type": "string",
                "description": "结构化地址，如：北京市朝阳区建国门外大街1号国贸大厦",
            },
            "city": {
                "type": "string",
                "description": "指定查询城市（可选），如：北京",
            },
        },
        "required": ["address"],
    }

    def __init__(self, api_key: str = ""):
        self._client = AmapClient(api_key)

    async def execute(self, address: str, city: str = "") -> str:
        params: dict[str, str] = {"address": address}
        if city:
            params["city"] = city

        data = await self._client.get("/geocode/geo", params)
        if data["status"] != "1":
            return f"地理编码失败: {data['info']}"

        geocodes = data.get("geocodes", [])
        if not geocodes:
            return f"未找到地址: {address}"

        g = geocodes[0]
        return (
            f"  地址: {g.get('formatted_address', address)}\n"
            f"  经纬度: {g.get('location', '未知')}\n"
            f"  区域: {g.get('country', '')} {g.get('province', '')} {g.get('city', '')} {g.get('district', '')}\n"
            f"  级别: {g.get('level', '')}"
        )


# ── 2. 逆地理编码 ────────────────────────────────────

class AmapRegeo(BaseTool):
    """逆地理编码：将经纬度坐标转换为结构化地址"""

    name = "amap_regeo"
    description = "将经纬度坐标转换为详细地址（高德地图逆地理编码）"
    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "经纬度坐标，格式：经度,纬度（如 116.397128,39.916527）",
            },
            "poi_types": {
                "type": "string",
                "description": "需要返回的POI类型（可选），如：餐饮服务|购物服务",
            },
        },
        "required": ["location"],
    }

    def __init__(self, api_key: str = ""):
        self._client = AmapClient(api_key)

    async def execute(self, location: str, poi_types: str = "") -> str:
        params: dict[str, str] = {"location": location, "extensions": "all"}
        if poi_types:
            params["poitype"] = poi_types

        data = await self._client.get("/geocode/regeo", params)
        if data["status"] != "1":
            return f"逆地理编码失败: {data['info']}"

        regeo = data.get("regeocode", {})
        addr = regeo.get("formatted_address", "未知地址")

        # 周边 POI
        pois = regeo.get("pois", [])[:5]
        poi_lines = []
        for p in pois:
            poi_lines.append(f"    · {p.get('name', '')} — {p.get('type', '')}")

        lines = [
            f"  地址: {addr}",
        ]
        addr_component = regeo.get("addressComponent", {})
        if addr_component:
            lines.append(
                f"  区域: {addr_component.get('country', '')} "
                f"{addr_component.get('province', '')} "
                f"{addr_component.get('city', '')} "
                f"{addr_component.get('district', '')}"
            )
            lines.append(f"  邮编: {addr_component.get('adcode', '')}")

        if poi_lines:
            lines.append("  周边 POI:")
            lines.extend(poi_lines)

        return "\n".join(lines)


# ── 3. 天气查询 ──────────────────────────────────────

class AmapWeather(BaseTool):
    """天气查询：按城市名/adcode 查询实时天气或预报"""

    name = "amap_weather"
    description = "查询指定城市的实时天气（高德地图），支持中文城市名"
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名或 adcode（行政区编码），如：北京、上海、深圳、110000",
            },
            "forecast": {
                "type": "boolean",
                "description": "是否查询未来3天预报，默认只查实时天气",
            },
        },
        "required": ["city"],
    }

    def __init__(self, api_key: str = ""):
        self._client = AmapClient(api_key)

    async def execute(self, city: str, forecast: bool = False) -> str:
        params: dict[str, Any] = {
            "city": city,
            "extensions": "all" if forecast else "base",
        }

        data = await self._client.get("/weather/weatherInfo", params)
        if data["status"] != "1":
            return f"天气查询失败: {data['info']}"

        if forecast:
            return self._format_forecast(data)
        return self._format_live(data)

    @staticmethod
    def _format_live(data: dict) -> str:
        lives = data.get("lives", [])
        if not lives:
            return "暂无天气数据"
        w = lives[0]
        return (
            f"  📍 {w.get('city', '')} ({w.get('adcode', '')})\n"
            f"  🌤 {w.get('weather', '')}\n"
            f"  🌡 {w.get('temperature', '')}°C（体感 {w.get('temperature_float', '')}°C）\n"
            f"  💧 湿度 {w.get('humidity', '')}%\n"
            f"  💨 {w.get('winddirection', '')}风 {w.get('windpower', '')}级\n"
            f"  🕐 更新时间: {w.get('reporttime', '')}"
        )

    @staticmethod
    def _format_forecast(data: dict) -> str:
        forecasts = data.get("forecasts", [])
        if not forecasts:
            return "暂无预报数据"
        f = forecasts[0]
        lines = [
            f"  📍 {f.get('city', '')} ({f.get('adcode', '')})",
            f"  🕐 发布时间: {f.get('reporttime', '')}",
        ]
        for cast in f.get("casts", []):
            lines.append(
                f"\n  📅 {cast.get('date', '')} ({cast.get('week', '')})"
                f"\n     白天: {cast.get('dayweather', '')}  "
                f"{cast.get('daytemp_float', '')}°C  "
                f"{cast.get('daywind', '')}风 {cast.get('daypower', '')}级"
                f"\n     夜间: {cast.get('nightweather', '')}  "
                f"{cast.get('nighttemp_float', '')}°C  "
                f"{cast.get('nightwind', '')}风 {cast.get('nightpower', '')}级"
            )
        return "\n".join(lines)


# ── 4. POI 搜索 ──────────────────────────────────────

class AmapTextSearch(BaseTool):
    """POI 关键词搜索"""

    name = "amap_text_search"
    description = "按关键词搜索POI（兴趣点），如：餐厅、酒店、加油站等"
    parameters = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "搜索关键词，如：星巴克、北京烤鸭、加油站",
            },
            "city": {
                "type": "string",
                "description": "城市名（可选），如：北京",
            },
            "types": {
                "type": "string",
                "description": "POI类型（可选），如：餐饮服务|购物服务",
            },
            "page": {
                "type": "integer",
                "description": "页码，默认1",
            },
            "page_size": {
                "type": "integer",
                "description": "每页条数，默认10，最大25",
            },
        },
        "required": ["keywords"],
    }

    def __init__(self, api_key: str = ""):
        self._client = AmapClient(api_key)

    async def execute(
        self,
        keywords: str,
        city: str = "",
        types: str = "",
        page: int = 1,
        page_size: int = 10,
    ) -> str:
        params: dict[str, Any] = {
            "keywords": keywords,
            "offset": min(page_size, 25),
            "page": page,
            "extensions": "base",
        }
        if city:
            params["city"] = city
        if types:
            params["types"] = types

        data = await self._client.get("/place/text", params)
        if data["status"] != "1":
            return f"POI搜索失败: {data['info']}"

        pois = data.get("pois", [])
        if not pois:
            return f"未找到 '{keywords}' 相关结果"

        count = data.get("count", "0")
        lines = [f"  找到 {count} 个 '{keywords}' 的结果（第{page}页）:"]
        for idx, p in enumerate(pois[:page_size], 1):
            lines.append(
                f"\n  {idx}. {p.get('name', '')}"
                f"\n     地址: {p.get('address', '')}"
                f"\n     类型: {p.get('type', '')}"
                f"\n     坐标: {p.get('location', '')}"
                f"\n     电话: {p.get('tel', '无')}"
            )

        return "\n".join(lines)


class AmapAroundSearch(BaseTool):
    """POI 周边搜索"""

    name = "amap_around_search"
    description = "搜索指定坐标周边的POI（兴趣点），如：附近的餐厅、酒店"
    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "中心点经纬度，格式：经度,纬度",
            },
            "keywords": {
                "type": "string",
                "description": "搜索关键词（可选）",
            },
            "radius": {
                "type": "integer",
                "description": "搜索半径（米），默认1000，最大50000",
            },
            "types": {
                "type": "string",
                "description": "POI类型（可选），如：餐饮服务|购物服务",
            },
            "page": {
                "type": "integer",
                "description": "页码，默认1",
            },
            "page_size": {
                "type": "integer",
                "description": "每页条数，默认25，最大25",
            },
        },
        "required": ["location"],
    }

    def __init__(self, api_key: str = ""):
        self._client = AmapClient(api_key)

    async def execute(
        self,
        location: str,
        keywords: str = "",
        radius: int = 1000,
        types: str = "",
        page: int = 1,
        page_size: int = 25,
    ) -> str:
        params: dict[str, Any] = {
            "location": location,
            "radius": min(radius, 50000),
            "offset": min(page_size, 25),
            "page": page,
            "extensions": "base",
        }
        if keywords:
            params["keywords"] = keywords
        if types:
            params["types"] = types

        data = await self._client.get("/place/around", params)
        if data["status"] != "1":
            return f"周边搜索失败: {data['info']}"

        pois = data.get("pois", [])
        if not pois:
            return f"坐标 {location} 周边 {radius} 米范围内未找到结果"

        count = data.get("count", "0")
        lines = [f"  坐标 {location} 周边 {radius} 米范围内找到 {count} 个结果:"]
        for idx, p in enumerate(pois[:15], 1):
            dist = p.get("distance", "?")
            lines.append(
                f"\n  {idx}. {p.get('name', '')}（距中心 {dist}米）"
                f"\n     地址: {p.get('address', '')}"
                f"\n     类型: {p.get('type', '')}"
                f"\n     电话: {p.get('tel', '无')}"
            )

        return "\n".join(lines)


class AmapSearchDetail(BaseTool):
    """POI 详情查询"""

    name = "amap_search_detail"
    description = "根据 POI ID 查询兴趣点详细信息"
    parameters = {
        "type": "object",
        "properties": {
            "poi_id": {
                "type": "string",
                "description": "POI 的唯一标识 ID（来自文本搜索或周边搜索结果）",
            },
        },
        "required": ["poi_id"],
    }

    def __init__(self, api_key: str = ""):
        self._client = AmapClient(api_key)

    async def execute(self, poi_id: str) -> str:
        data = await self._client.get("/place/detail", {"id": poi_id})
        if data["status"] != "1":
            return f"查询失败: {data['info']}"

        pois = data.get("pois", [])
        if not pois:
            return f"未找到 POI ID: {poi_id}"

        p = pois[0]
        return (
            f"  📍 {p.get('name', '')}\n"
            f"  类型: {p.get('type', '')}\n"
            f"  地址: {p.get('address', '')}\n"
            f"  坐标: {p.get('location', '')}\n"
            f"  电话: {p.get('tel', '无')}\n"
            f"  网站: {p.get('website', '无')}\n"
            f"  营业时间: {p.get('business_area', '')}\n"
            f"  评分: {p.get('biz_ext', {}).get('rating', '无')}"
        )


# ── 5. 路线规划 ──────────────────────────────────────

class AmapDirection(BaseTool):
    """驾车路线规划"""

    name = "amap_direction"
    description = "驾车路线规划（高德地图），计算起点到终点的驾车路线"
    parameters = {
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "起点经纬度，格式：经度,纬度",
            },
            "destination": {
                "type": "string",
                "description": "终点经纬度，格式：经度,纬度",
            },
            "strategy": {
                "type": "integer",
                "description": "驾车策略：0=速度最快（默认），1=费用最少，2=距离最短，3=不走高速，4=躲避拥堵",
            },
        },
        "required": ["origin", "destination"],
    }

    def __init__(self, api_key: str = ""):
        self._client = AmapClient(api_key)

    async def execute(self, origin: str, destination: str, strategy: int = 0) -> str:
        data = await self._client.get("/direction/driving", {
            "origin": origin,
            "destination": destination,
            "strategy": strategy,
            "extensions": "base",
        })
        if data["status"] != "1":
            return f"路线规划失败: {data['info']}"

        route = data.get("route", {})
        paths = route.get("paths", [])
        if not paths:
            return "未找到可行路线"

        p = paths[0]
        steps = p.get("steps", [])
        step_lines = []
        for i, step in enumerate(steps[:10], 1):
            instruction = step.get("instruction", "")
            road = step.get("road", "")
            distance = step.get("distance", "0")
            toll = step.get("toll", "0")
            step_lines.append(
                f"    {i}. {instruction}"
                f"{' (' + road + ')' if road else ''}"
                f" — {distance}米"
                f" {'(收费' + toll + '元)' if toll != '0' else ''}"
            )

        if len(steps) > 10:
            step_lines.append(f"    ... 共 {len(steps)} 个导航步骤")

        strategy_names = ["速度最快", "费用最少", "距离最短", "不走高速", "躲避拥堵"]
        return (
            f"  🚗 驾车路线规划 — {strategy_names[strategy] if strategy < len(strategy_names) else '自定义'}\n"
            f"  总距离: {round(int(p.get('distance', 0)) / 1000, 1)} km\n"
            f"  预计时间: {round(int(p.get('duration', 0)) / 60, 0)} 分钟\n"
            f"  收费: {p.get('toll', '0')} 元\n"
            f"  导航步骤:\n" + "\n".join(step_lines)
        )


# ── 6. 距离测量 ──────────────────────────────────────

class AmapDistance(BaseTool):
    """距离测量：计算两个坐标间的直线距离"""

    name = "amap_distance"
    description = "计算两个经纬度坐标之间的直线距离（高德地图）"
    parameters = {
        "type": "object",
        "properties": {
            "origins": {
                "type": "string",
                "description": "起点坐标，格式：经度,纬度",
            },
            "destination": {
                "type": "string",
                "description": "终点坐标，格式：经度,纬度",
            },
            "type": {
                "type": "integer",
                "description": "距离类型：0=直线距离（默认），1=驾车距离",
            },
        },
        "required": ["origins", "destination"],
    }

    def __init__(self, api_key: str = ""):
        self._client = AmapClient(api_key)

    async def execute(self, origins: str, destination: str, type: int = 0) -> str:
        data = await self._client.get("/distance", {
            "origins": origins,
            "destination": destination,
            "type": type,
        })
        if data["status"] != "1":
            return f"距离测量失败: {data['info']}"

        results = data.get("results", [])
        if not results:
            return "未获取到距离数据"

        r = results[0]
        dist = int(r.get("distance", 0))
        mode = "直线" if type == 0 else "驾车"
        if dist >= 1000:
            return f"  📏 {mode}距离: {round(dist / 1000, 2)} km"
        return f"  📏 {mode}距离: {dist} 米"


# ── 导出所有工具 ──────────────────────────────────────

ALL_AMAP_TOOLS = [
    AmapGeo,
    AmapRegeo,
    AmapWeather,
    AmapTextSearch,
    AmapAroundSearch,
    AmapSearchDetail,
    AmapDirection,
    AmapDistance,
]


def register_all_amap_tools(
    registry: "ToolRegistry",
    api_key: str = "",
) -> int:
    """注册所有高德地图工具到 ToolRegistry

    参数:
        registry: ToolRegistry 实例
        api_key:  高德 API Key（可选，默认从环境变量读取）

    返回:
        成功注册的工具数量
    """
    count = 0
    for tool_cls in ALL_AMAP_TOOLS:
        try:
            tool = tool_cls(api_key) if api_key else tool_cls()
            registry.register(tool)
            count += 1
        except ValueError as e:
            logger.error("注册 %s 失败: %s", tool_cls.__name__, e)
            return count  # API Key 无效，停止注册后续工具
    return count
